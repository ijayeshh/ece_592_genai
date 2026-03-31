"""Document loading via LangChain loaders."""

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> list[Document]:
    """Load all .txt files from *data_dir* using DirectoryLoader + TextLoader.

    Each returned Document has:
        metadata["source"]  – absolute file path
        metadata["doc_id"]  – basename of the file (e.g. 'USFED_FF_STAT_2018.txt')
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    logger.info("Loading .txt documents from: %s", data_path)

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=False,
    )
    docs = loader.load()

    # Normalise metadata
    for doc in docs:
        source = doc.metadata.get("source", "")
        doc.metadata["source"] = source
        doc.metadata["doc_id"] = os.path.basename(source)

    logger.info("Loaded %d document(s).", len(docs))
    return docs
