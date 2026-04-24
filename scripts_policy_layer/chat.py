"""Interactive chat loop using the policy-layer RAG pipeline."""

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
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("chat")

HELP_TEXT = """\
Commands:
  :exit            – quit
  :debug on        – enable debug output (chunks + metadata + prompt)
  :debug off       – disable debug output
  :jurisdiction    – show active jurisdiction filter
  :help            – show this help
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RAG chat (policy layer).")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--persist_dir", default=".chroma_langchain_policy")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--llm_backend", default="groq", choices=["groq", "huggingface"])
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
        help="Restrict retrieved chunks to a single jurisdiction.",
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="Disable authority-rank reranking.",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable temporal deduplication.",
    )

    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def print_result(result: dict, debug: bool) -> None:
    if debug:
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
        print("\n--- Context ---")
        print(result["context_text"])
        print("\n--- Full Prompt ---")
        print(result["final_prompt"])

    print("\nANSWER:", result["answer"])
    print()


def main() -> None:
    args = parse_args()
    debug = args.debug

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
        debug=debug,
    )

    print("Loading RAG pipeline…")
    pipeline = RAGPipeline(config)

    active = []
    if config.jurisdiction_filter:
        active.append(f"jurisdiction={config.jurisdiction_filter}")
    if config.apply_authority_rerank:
        active.append("authority-rerank=ON")
    if config.apply_temporal_dedup:
        active.append("temporal-dedup=ON")
    filter_str = ", ".join(active) if active else "none"
    print(f"Active policy filters: {filter_str}")
    print("Ready. Type your question or a command (:help for commands).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() == ":exit":
            print("Goodbye!")
            sys.exit(0)
        elif user_input.lower() == ":help":
            print(HELP_TEXT)
            continue
        elif user_input.lower() == ":debug on":
            debug = True
            print("[Debug mode ON]")
            continue
        elif user_input.lower() == ":debug off":
            debug = False
            print("[Debug mode OFF]")
            continue
        elif user_input.lower() == ":jurisdiction":
            print(f"[Jurisdiction filter: {config.jurisdiction_filter or 'none (both jurisdictions)'}]")
            continue
        elif user_input.startswith(":"):
            print(f"Unknown command: {user_input}. Type :help for commands.")
            continue

        try:
            result = pipeline.answer(user_input)
            print_result(result, debug)
        except Exception as exc:  # noqa: BLE001
            print(f"[Error] {exc}")


if __name__ == "__main__":
    main()
