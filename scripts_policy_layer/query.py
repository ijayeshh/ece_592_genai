"""One-shot query runner for the policy-layer RAG pipeline."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_langchain_policy._env import load_env
load_env()

from rag_langchain_policy.config import RAGConfig
from rag_langchain_policy.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single RAG query (policy layer).")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--persist_dir", default=".chroma_langchain_policy")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--llm_backend", default="groq", choices=["groq", "huggingface"],
    )
    parser.add_argument("--llm_model", default="llama-3.1-8b-instant")
    parser.add_argument("--hf_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)

    # ── Policy filter args ────────────────────────────────────────────────────
    parser.add_argument(
        "--jurisdiction",
        default=None,
        choices=["US-FED", "US-CA"],
        help="Restrict retrieved chunks to a single jurisdiction. "
             "Default: no filter (both jurisdictions).",
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="Disable authority-rank reranking (statutes vs guidance).",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable temporal deduplication (keep latest doc per topic).",
    )

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RAGConfig(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        hf_model=args.hf_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # Policy filters
        jurisdiction_filter=args.jurisdiction,
        apply_authority_rerank=not args.no_rerank,
        apply_temporal_dedup=not args.no_dedup,
        debug=args.debug,
    )

    pipeline = RAGPipeline(config)
    result = pipeline.answer(args.query)

    print("\n" + "=" * 70)
    print(f"QUERY: {args.query}")
    filters_active = []
    if args.jurisdiction:
        filters_active.append(f"jurisdiction={args.jurisdiction}")
    if not args.no_rerank:
        filters_active.append("authority-rerank=ON")
    if not args.no_dedup:
        filters_active.append("temporal-dedup=ON")
    if filters_active:
        print(f"FILTERS: {', '.join(filters_active)}")
    print("=" * 70)

    if args.debug:
        print("\n--- Retrieved Chunks (after policy filters) ---")
        for chunk in result["retrieved_chunks"]:
            print(
                f"  [{chunk['rank']}] {chunk['doc_id']}:{chunk['chunk_index']} "
                f"(score={chunk['score']:.4f}) "
                f"| {chunk.get('jurisdiction','?')} "
                f"| {chunk.get('effective_date','?')} "
                f"| authority_rank={chunk.get('authority_rank','?')}"
            )
            print(f"      {chunk['preview']}")
        print("\n--- Context Fed to LLM ---")
        print(result["context_text"])
        print("\n--- Full Prompt ---")
        print(result["final_prompt"])

    print("\n--- ANSWER ---")
    print(result["answer"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
