"""
API key + common dependencies for FastAPI routes.

X-API-Key header is checked against the NIDS_API_KEY environment variable.
If NIDS_API_KEY is not set (dev / local), the check is bypassed so the API
works out-of-the-box without configuration.  Set it in .env for production.
"""
import logging
import os

from fastapi import Header, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_API_KEY: str | None = os.environ.get("NIDS_API_KEY", "").strip() or None

# Declare the security scheme for OpenAPI docs (shows padlock icon)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency that enforces X-API-Key authentication.

    - If NIDS_API_KEY env var is not set: key check is bypassed (dev-friendly).
    - If NIDS_API_KEY is set: the header must match exactly or 403 is returned.
    - The key value is never logged or surfaced in error responses.
    """
    if _API_KEY is None:
        # Dev mode — no key configured, allow all
        return
    if x_api_key != _API_KEY:
        logger.warning("Rejected request: invalid or missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )
