"""Single source of truth for all RAG baseline configuration."""

from dataclasses import dataclass


@dataclass
class RAGConfig:
    # Data
    data_dir: str = "synthetic_data_legal"
    persist_dir: str = ".chroma_langchain_baseline"

    # Chunking
    chunk_size: int = 350
    chunk_overlap: int = 60

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval
    top_k: int = 5

    # LLM backend: "groq" | "huggingface"
    llm_backend: str = "groq"

    # Groq model (used when llm_backend="groq")
    llm_model: str = "llama-3.1-8b-instant"

    # HuggingFace generation settings (used when llm_backend="huggingface")
    hf_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True

    # Debug
    debug: bool = False
