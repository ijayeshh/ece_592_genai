"""Retriever construction and result formatting."""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def build_retriever(vectorstore: Chroma, top_k: int = 5) -> BaseRetriever:
    """Return a similarity-search retriever for *top_k* chunks."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    logger.debug("Retriever built with top_k=%d.", top_k)
    return retriever


def retrieve_with_scores(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Return (Document, score) pairs for *query* using similarity search."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    return results


def format_retrieved_chunks(
    docs_and_scores: list[tuple[Document, float]],
) -> list[dict]:
    """Convert raw retrieval results into serialisable dicts.

    Each dict now also carries jurisdiction, effective_date, authority_rank
    when present — ready for the policy filtering layer.
    """
    formatted = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        formatted.append(
            {
                "rank": rank,
                "doc_id": meta.get("doc_id", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "score": round(float(score), 4),
                # Policy metadata (present when index was built with manifest)
                "jurisdiction": meta.get("jurisdiction"),
                "effective_date": meta.get("effective_date"),
                "authority_rank": meta.get("authority_rank"),
                "preview": doc.page_content[:200].strip(),
                "text": doc.page_content,
            }
        )
    return formatted
