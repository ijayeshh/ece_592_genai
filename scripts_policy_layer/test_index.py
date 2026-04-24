"""Quick sanity-check for the policy-layer Chroma index.

Verifies:
  1. Index loads without error
  2. Correct number of chunks stored
  3. Every chunk has doc_id + chunk_index
  4. Every chunk has jurisdiction, effective_date, authority_rank
  5. A sample similarity search returns results with metadata
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_langchain_policy._env import load_env
load_env()

from rag_langchain_policy.vectorstore import load_vectorstore

PERSIST_DIR = ".chroma_langchain_policy"
EXPECTED_CHUNKS = None  # set to an int after first build to assert exact count
POLICY_KEYS = ("jurisdiction", "effective_date", "authority_rank")

def run_tests():
    print("=" * 60)
    print("Policy Index Sanity Check")
    print("=" * 60)

    # ── 1. Load index ──────────────────────────────────────────────
    print("\n[1] Loading Chroma index...")
    vs = load_vectorstore(PERSIST_DIR)
    print("    OK — index loaded")

    # ── 2. Chunk count ─────────────────────────────────────────────
    print(f"\n[2] Checking chunk count (expected {EXPECTED_CHUNKS})...")
    count = vs._collection.count()
    if EXPECTED_CHUNKS is not None:
        assert count == EXPECTED_CHUNKS, f"Expected {EXPECTED_CHUNKS} chunks, got {count}"
    print(f"    OK — {count} chunks found (chunk_size=700)")

    # ── 3 & 4. Metadata on every chunk ────────────────────────────
    print("\n[3] Checking metadata on all chunks...")
    results = vs._collection.get(include=["metadatas"])
    metadatas = results["metadatas"]

    missing_base, missing_policy = [], []
    for i, meta in enumerate(metadatas):
        if not meta.get("doc_id") or meta.get("chunk_index") is None:
            missing_base.append(i)
        for key in POLICY_KEYS:
            if key not in meta:
                missing_policy.append((i, key))

    assert not missing_base, f"Chunks missing doc_id/chunk_index: {missing_base}"
    print(f"    OK — all {len(metadatas)} chunks have doc_id + chunk_index")

    assert not missing_policy, f"Chunks missing policy keys: {missing_policy[:5]}"
    print(f"    OK — all chunks have {POLICY_KEYS}")

    # ── 5. Sample similarity search ────────────────────────────────
    print("\n[4] Running sample similarity search...")
    hits = vs.similarity_search_with_relevance_scores(
        "penalty for financial fraud", k=3
    )
    assert len(hits) == 3, f"Expected 3 hits, got {len(hits)}"
    print(f"    OK — got {len(hits)} results")
    print()
    for rank, (doc, score) in enumerate(hits, 1):
        m = doc.metadata
        print(
            f"    [{rank}] {m['doc_id']}:{m['chunk_index']} "
            f"score={score:.4f} | {m['jurisdiction']} "
            f"| {m['effective_date']} | authority_rank={m['authority_rank']}"
        )
        print(f"        {doc.page_content[:120].strip()}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
