"""Interactive chat loop using the RAG baseline pipeline."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_langchain_baseline._env import load_env
load_env()

from rag_langchain_baseline.config import RAGConfig
from rag_langchain_baseline.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("chat")

HELP_TEXT = """\
Commands:
  :exit          – quit
  :debug on      – enable debug output (retrieved chunks + prompt)
  :debug off     – disable debug output
  :help          – show this help
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RAG chat.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--persist_dir", default=".chroma_langchain_baseline")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--llm_backend",
        default="groq",
        choices=["groq", "huggingface"],
        help="LLM backend to use (default: groq)",
    )
    parser.add_argument(
        "--llm_model",
        default="llama-3.1-8b-instant",
        help="Groq model name (used when --llm_backend groq)",
    )
    parser.add_argument(
        "--hf_model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model ID (used when --llm_backend huggingface)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def print_result(result: dict, debug: bool) -> None:
    if debug:
        print("\n--- Retrieved Chunks ---")
        for chunk in result["retrieved_chunks"]:
            print(
                f"  [{chunk['rank']}] {chunk['doc_id']}:{chunk['chunk_index']} "
                f"(score={chunk['score']:.4f})"
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
        debug=debug,
    )

    print("Loading RAG pipeline…")
    pipeline = RAGPipeline(config)
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
