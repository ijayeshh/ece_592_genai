"""Load .env into the environment as early as possible.

Call load_env() once at the top of any entry-point script.
Uses python-dotenv; if the .env file is missing the call is a no-op.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root is two levels up from this file (rag_langchain_policy/_env.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _REPO_ROOT / ".env"


def load_env() -> None:
    """Load .env from the repo root and set HF_TOKEN if present."""
    try:
        from dotenv import load_dotenv

        loaded = load_dotenv(_ENV_FILE, override=False)
        if loaded:
            logger.debug(".env loaded from %s", _ENV_FILE)
        else:
            logger.debug(".env not found at %s — skipping.", _ENV_FILE)
    except ImportError:
        logger.warning("python-dotenv not installed; .env file will not be loaded.")

    _apply_hf_token()


def _apply_hf_token() -> None:
    """Push HF_TOKEN into the HuggingFace hub login if set."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or "xxx" in token.lower():
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logger.debug("HuggingFace hub login successful.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("HuggingFace hub login failed: %s", exc)
