"""API-key simulation for the prototype (Section 5.1)."""

from fastapi import Header, HTTPException

# In production this would validate against a real IdP.
_VALID_KEYS = {"demo-key-001", "demo-key-002", "test-key"}


async def verify_api_key(x_api_key: str = Header(default="demo-key-001")) -> str:
    """FastAPI dependency that validates the caller's API key."""
    if x_api_key not in _VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
