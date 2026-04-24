"""Build (or rebuild) the metadata-enriched Chroma index from the legal corpus."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_langchain_policy._env import load_env
load_env()

from rag_langchain_policy.config import RAGConfig
from rag_langchain_policy.loaders import load_documents
from rag_langchain_policy.splitting import split_documents
from rag_langchain_policy.vectorstore import build_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata-enriched Chroma index from .txt documents."
    )
    parser.add_argument("--data_dir", default="synthetic_data_legal")
    parser.add_argument(
        "--metadata_path",
        default="synthetic_data_legal/metadata_manifest.csv",
        help="Path to metadata_manifest.csv. Pass empty string to skip.",
    )
    parser.add_argument("--persist_dir", default=".chroma_langchain_policy")
    parser.add_argument("--chunk_size", type=int, default=700)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Treat empty string as "no manifest"
    metadata_path = args.metadata_path.strip() or None

    config = RAGConfig(
        data_dir=args.data_dir,
        metadata_path=metadata_path or RAGConfig.metadata_path,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )

    logger.info("=== Building RAG index (policy layer) ===")
    logger.info("  data_dir       : %s", config.data_dir)
    logger.info("  metadata_path  : %s", metadata_path or "(none)")
    logger.info("  persist_dir    : %s", config.persist_dir)
    logger.info("  chunk_size     : %d", config.chunk_size)
    logger.info("  chunk_overlap  : %d", config.chunk_overlap)
    logger.info("  embedding_model: %s", config.embedding_model)

    docs = load_documents(config.data_dir, metadata_path)
    chunks = split_documents(
        docs,
        config.chunk_size,
        config.chunk_overlap,
        validate_metadata=bool(metadata_path),
    )
    build_vectorstore(chunks, config.persist_dir, config.embedding_model)

    logger.info("=== Index built successfully ===")
    logger.info("  Documents : %d", len(docs))
    logger.info("  Chunks    : %d", len(chunks))
    logger.info("  Persisted : %s", config.persist_dir)


if __name__ == "__main__":
    main()
