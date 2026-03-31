"""Build (or rebuild) the Chroma vector index from the legal .txt corpus."""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_langchain_baseline._env import load_env
load_env()

from rag_langchain_baseline.config import RAGConfig
from rag_langchain_baseline.loaders import load_documents
from rag_langchain_baseline.splitting import split_documents
from rag_langchain_baseline.vectorstore import build_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Chroma vector index from .txt documents."
    )
    parser.add_argument("--data_dir", default="synthetic_data_legal")
    parser.add_argument("--persist_dir", default=".chroma_langchain_baseline")
    parser.add_argument("--chunk_size", type=int, default=350)
    parser.add_argument("--chunk_overlap", type=int, default=60)
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RAGConfig(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )

    logger.info("=== Building RAG index ===")
    logger.info("  data_dir      : %s", config.data_dir)
    logger.info("  persist_dir   : %s", config.persist_dir)
    logger.info("  chunk_size    : %d", config.chunk_size)
    logger.info("  chunk_overlap : %d", config.chunk_overlap)
    logger.info("  embedding_model: %s", config.embedding_model)

    docs = load_documents(config.data_dir)
    chunks = split_documents(docs, config.chunk_size, config.chunk_overlap)
    build_vectorstore(chunks, config.persist_dir, config.embedding_model)

    logger.info("=== Index built successfully ===")
    logger.info("  Documents : %d", len(docs))
    logger.info("  Chunks    : %d", len(chunks))
    logger.info("  Persisted : %s", config.persist_dir)


if __name__ == "__main__":
    main()
