"""Chroma vector store management."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    """Return a LangChain HuggingFaceEmbeddings object."""
    device = _get_device()
    logger.info("Loading embedding model: %s (device=%s)", model_name, device)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def build_vectorstore(
    chunks: list[Document],
    persist_dir: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> Chroma:
    """Embed *chunks* and persist a Chroma collection to *persist_dir*."""
    persist_path = Path(persist_dir).resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(embedding_model)

    logger.info(
        "Building Chroma index with %d chunks → %s", len(chunks), persist_path
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name="rag_policy",
    )
    logger.info("Chroma index built and persisted.")
    return vectorstore


def load_vectorstore(
    persist_dir: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> Chroma:
    """Load an existing Chroma collection from *persist_dir*."""
    persist_path = Path(persist_dir).resolve()
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma persist directory not found: {persist_path}. "
            "Run scripts_policy_layer/build_index.py first."
        )

    embeddings = get_embeddings(embedding_model)

    logger.info("Loading Chroma index from: %s", persist_path)
    vectorstore = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name="rag_policy",
    )
    return vectorstore
