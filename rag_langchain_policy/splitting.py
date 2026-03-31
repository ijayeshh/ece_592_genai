"""Text chunking via LangChain RecursiveCharacterTextSplitter."""

import logging
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loaders import MANIFEST_KEYS

logger = logging.getLogger(__name__)


def split_documents(
    docs: list[Document],
    chunk_size: int = 350,
    chunk_overlap: int = 60,
    validate_metadata: bool = False,
) -> list[Document]:
    """Split *docs* into chunks and tag each chunk with *chunk_index*.

    chunk_index is 0-based per doc_id so that citations are stable.

    If *validate_metadata* is True, every chunk is asserted to carry the three
    policy metadata keys (jurisdiction, effective_date, authority_rank).
    Pass this flag whenever a manifest was provided during loading — it catches
    missing or mismatched manifest rows before they reach the vector store.
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

    # Optional early validation — fail loudly rather than silently index bad data
    if validate_metadata:
        bad: list[str] = []
        for chunk in chunks:
            missing = [k for k in MANIFEST_KEYS if k not in chunk.metadata]
            if missing:
                bad.append(
                    f"{chunk.metadata.get('doc_id','?')}:"
                    f"{chunk.metadata.get('chunk_index','?')} missing {missing}"
                )
        if bad:
            raise ValueError(
                f"Metadata validation failed for {len(bad)} chunk(s):\n"
                + "\n".join(bad[:10])
                + ("\n  …" if len(bad) > 10 else "")
            )
        logger.info("Metadata validation passed for all %d chunks.", len(chunks))

    logger.info(
        "Split %d document(s) into %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d).",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
