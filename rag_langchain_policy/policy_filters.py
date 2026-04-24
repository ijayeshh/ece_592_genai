"""Post-retrieval policy filters.

Three filters applied in order to the raw similarity-search results:

  1. Jurisdiction filter  — discard chunks not matching the requested jurisdiction
  2. Authority rerank     — prefer statutes (rank 3) over guidance/FAQ (rank 2)
  3. Temporal dedup       — for the same topic+jurisdiction, keep only the most
                            recent document's chunks

All three operate on the list[dict] produced by retriever.format_retrieved_chunks().
No Chroma re-queries are needed.
"""

import logging
from datetime import date

logger = logging.getLogger(__name__)

# Filename convention: PREFIX_Topic_Type_YEAR.txt
# Topic stem = everything before the last underscore-delimited token that is a year.
# e.g.  USFED_FinancialFraud_Statute_2018.txt  →  USFED_FinancialFraud_Statute
#        USCA_AG_Bulletin_2020.txt               →  USCA_AG_Bulletin
def _topic_stem(doc_id: str) -> str:
    """Strip trailing _YEAR and .txt from a doc_id to get the topic stem."""
    name = doc_id.removesuffix(".txt")
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return name


def filter_by_jurisdiction(
    chunks: list[dict],
    jurisdiction: str,
) -> list[dict]:
    """Keep only chunks whose jurisdiction matches *jurisdiction* exactly.

    e.g. jurisdiction="US-FED" removes all US-CA chunks and vice-versa.
    """
    filtered = [c for c in chunks if c.get("jurisdiction") == jurisdiction]
    logger.debug(
        "Jurisdiction filter (%s): %d → %d chunks", jurisdiction, len(chunks), len(filtered)
    )
    return filtered


def rerank_by_authority(chunks: list[dict]) -> list[dict]:
    """Sort chunks by (-authority_rank, -score).

    Effect: among chunks with similar similarity scores, statutes (rank=3)
    are always placed above guidance/FAQ documents (rank=2).
    Chunks with no authority_rank are placed last.
    """
    def sort_key(c: dict) -> tuple:
        rank = c.get("authority_rank") or 0
        score = c.get("score") or 0.0
        return (-rank, -score)

    reranked = sorted(chunks, key=sort_key)
    logger.debug("Authority rerank applied to %d chunks.", len(reranked))
    return reranked


def deduplicate_by_recency(chunks: list[dict]) -> list[dict]:
    """For each (jurisdiction, authority_rank, topic_stem) group, keep only
    chunks from the document with the most recent effective_date.

    This ensures that when both a 2018 statute and a 2024 statute for the same
    topic are retrieved, only the 2024 version's chunks appear in the context.
    """
    # Build a map: group_key → best effective_date seen so far
    best_date: dict[tuple, date] = {}

    for chunk in chunks:
        key = (
            chunk.get("jurisdiction"),
            chunk.get("authority_rank"),
            _topic_stem(chunk.get("doc_id", "")),
        )
        raw_date = chunk.get("effective_date") or "1900-01-01"
        try:
            parsed = date.fromisoformat(raw_date)
        except ValueError:
            parsed = date(1900, 1, 1)

        if key not in best_date or parsed > best_date[key]:
            best_date[key] = parsed

    # Keep only chunks that match their group's best date
    kept = []
    for chunk in chunks:
        key = (
            chunk.get("jurisdiction"),
            chunk.get("authority_rank"),
            _topic_stem(chunk.get("doc_id", "")),
        )
        raw_date = chunk.get("effective_date") or "1900-01-01"
        try:
            parsed = date.fromisoformat(raw_date)
        except ValueError:
            parsed = date(1900, 1, 1)

        if parsed == best_date[key]:
            kept.append(chunk)

    logger.debug("Temporal dedup: %d → %d chunks", len(chunks), len(kept))
    return kept


def apply_policy_filters(
    chunks: list[dict],
    top_k: int,
    jurisdiction_filter: str | None,
    apply_authority_rerank: bool,
    apply_temporal_dedup: bool,
) -> list[dict]:
    """Run the full post-retrieval policy filter stack and return top_k chunks.

    Order of operations:
      1. Jurisdiction filter (if jurisdiction_filter is set)
      2. Authority-rank rerank (if apply_authority_rerank is True)
      3. Temporal deduplication (if apply_temporal_dedup is True)
      4. Trim to top_k
      5. Re-assign 1-based rank field

    Args:
        chunks:                 Raw list from format_retrieved_chunks().
        top_k:                  Final number of chunks to return.
        jurisdiction_filter:    "US-FED", "US-CA", or None (no filter).
        apply_authority_rerank: Enable authority-rank secondary sort.
        apply_temporal_dedup:   Enable same-topic recency deduplication.

    Returns:
        Filtered, reranked, deduplicated list of at most top_k chunk dicts,
        with rank reassigned starting from 1.
    """
    original_count = len(chunks)

    if jurisdiction_filter:
        chunks = filter_by_jurisdiction(chunks, jurisdiction_filter)
        if not chunks:
            logger.warning(
                "Jurisdiction filter '%s' removed all %d candidates. "
                "Returning empty list.",
                jurisdiction_filter,
                original_count,
            )
            return []

    if apply_authority_rerank:
        chunks = rerank_by_authority(chunks)

    if apply_temporal_dedup:
        chunks = deduplicate_by_recency(chunks)

    # Trim to top_k
    chunks = chunks[:top_k]

    # Reassign rank (1-based) after all transformations
    for i, chunk in enumerate(chunks, start=1):
        chunk["rank"] = i

    logger.debug(
        "Policy filters done: %d → %d chunks (top_k=%d).",
        original_count,
        len(chunks),
        top_k,
    )
    return chunks
