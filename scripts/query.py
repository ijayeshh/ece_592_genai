"""One-shot query runner for the RAG baseline."""

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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single RAG query.")
    parser.add_argument(
        "--query",
        required=True,
        help='e.g. "How much jail time for financial fraud?"',
    )
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks, context, and full prompt.",
    )
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
        debug=args.debug,
    )

    pipeline = RAGPipeline(config)
    result = pipeline.answer(args.query)

    print("\n" + "=" * 70)
    print(f"QUERY: {args.query}")
    print("=" * 70)

    if args.debug:
        print("\n--- Retrieved Chunks ---")
        for chunk in result["retrieved_chunks"]:
            print(
                f"  [{chunk['rank']}] {chunk['doc_id']}:{chunk['chunk_index']} "
                f"(score={chunk['score']:.4f})"
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
