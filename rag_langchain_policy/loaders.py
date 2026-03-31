"""Document loading with optional metadata injection from a manifest CSV."""

import csv
import logging
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Keys that every document must carry when a manifest is provided
MANIFEST_KEYS = ("jurisdiction", "effective_date", "authority_rank")


def load_manifest(metadata_path: str) -> dict[str, dict]:
    """Read *metadata_path* CSV and return a dict keyed by doc_id.

    Expected CSV columns: doc_id, jurisdiction, effective_date, authority_rank

    Returns:
        {
            "USFED_FinancialFraud_Statute_2018.txt": {
                "jurisdiction": "Federal",
                "effective_date": "2018-01-01",
                "authority_rank": 1,
            },
            ...
        }
    """
    manifest_file = Path(metadata_path).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Metadata manifest not found: {manifest_file}")

    manifest: dict[str, dict] = {}
    with manifest_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            doc_id = row["doc_id"].strip()
            manifest[doc_id] = {
                "jurisdiction": row["jurisdiction"].strip(),
                "effective_date": row["effective_date"].strip(),
                "authority_rank": int(row["authority_rank"].strip()),
            }

    logger.info("Manifest loaded: %d entries from %s", len(manifest), manifest_file)
    return manifest


def load_documents(
    data_dir: str,
    metadata_path: str | None = None,
) -> list[Document]:
    """Load all .txt files from *data_dir* using DirectoryLoader + TextLoader.

    Each returned Document has at minimum:
        metadata["source"]   – absolute file path
        metadata["doc_id"]   – basename of the file

    If *metadata_path* is provided, three extra fields are attached per doc:
        metadata["jurisdiction"]    – e.g. "Federal" or "California"
        metadata["effective_date"]  – ISO date string, e.g. "2018-01-01"
        metadata["authority_rank"]  – int, 1 = statute (highest), 3 = FAQ (lowest)
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Load manifest first so we can fail fast before reading any documents
    manifest: dict[str, dict] = {}
    if metadata_path:
        manifest = load_manifest(metadata_path)

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

    missing_from_manifest: list[str] = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        doc_id = os.path.basename(source)
        doc.metadata["source"] = source
        doc.metadata["doc_id"] = doc_id

        if manifest:
            meta = manifest.get(doc_id)
            if meta:
                doc.metadata["jurisdiction"] = meta["jurisdiction"]
                doc.metadata["effective_date"] = meta["effective_date"]
                doc.metadata["authority_rank"] = meta["authority_rank"]
            else:
                missing_from_manifest.append(doc_id)

    if missing_from_manifest:
        logger.warning(
            "These documents have no manifest entry and will lack policy metadata: %s",
            missing_from_manifest,
        )

    logger.info(
        "Loaded %d document(s)%s.",
        len(docs),
        " with policy metadata" if manifest else "",
    )
    return docs
