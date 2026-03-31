"""Text chunking via LangChain RecursiveCharacterTextSplitter."""

import logging
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_documents(
    docs: list[Document],
    chunk_size: int = 350,
    chunk_overlap: int = 60,
) -> list[Document]:
    """Split *docs* into chunks and tag each chunk with *chunk_index*.

    chunk_index is 0-based per doc_id so that [doc_id:0], [doc_id:1], …
    matches the citation format used in prompts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)

    # Assign chunk_index per doc_id
    counter: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id", "unknown")
        chunk.metadata["chunk_index"] = counter[doc_id]
        counter[doc_id] += 1

    logger.info(
        "Split %d document(s) into %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d).",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
