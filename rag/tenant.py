"""
rag/tenant.py

Tenant context resolution for multi-tenant ControlPlane.ai deployments.

Each tenant gets:
  - Isolated vector store collections: {tenant_id}__{collection}
  - Isolated BM25 indexes per tenant
  - Per-tenant prompt version overrides
  - Per-tenant daily LLM budget limits

Tenant configs live in tenants/{tenant_id}.yaml.
Falls back to sensible defaults if config file not found.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_TENANTS_DIR = _ROOT / "tenants"

# Characters not allowed in tenant IDs (injection prevention)
_FORBIDDEN_CHARS = set(' /\\#?&=%+@!{}[]|<>"\'')


@dataclass
class TenantContext:
    """Resolved, immutable context object for a single tenant."""
    tenant_id: str
    display_name: str = "Default Tenant"
    # Prompt version overrides: {prompt_name: version_string}
    prompt_versions: dict = field(default_factory=dict)
    # LLM daily cost cap in USD (0.0 = no limit)
    daily_budget_usd: float = 50.0
    # Allowed departments for this tenant (empty = all allowed)
    allowed_departments: list = field(default_factory=list)
    # Whether tenant was explicitly configured via YAML
    is_configured: bool = False

    @property
    def namespace(self) -> str:
        """Prefix string for all vector collections. Empty for 'default' tenant."""
        if self.tenant_id == "default":
            return ""
        return f"{self.tenant_id}__"

    def namespaced_collection(self, collection_name: str) -> str:
        """Return collection name with tenant namespace applied."""
        if not self.namespace:
            return collection_name
        return f"{self.namespace}{collection_name}"

    def prompt_version_for(self, prompt_name: str, default: str = "latest") -> str:
        """Return this tenant's pinned version for a prompt, or the default."""
        return self.prompt_versions.get(prompt_name, default)


def validate_tenant_id(tenant_id: str) -> bool:
    """Return True iff tenant_id is a safe identifier string."""
    if not tenant_id or len(tenant_id) > 64:
        return False
    if any(c in _FORBIDDEN_CHARS for c in tenant_id):
        return False
    return True


def get_tenant_context(tenant_id: str = "default") -> TenantContext:
    """
    Load and return TenantContext for the given tenant_id.

    Looks for tenants/{tenant_id}.yaml. Falls back to a default
    TenantContext if the file is not found. Never raises.
    """
    if not validate_tenant_id(tenant_id):
        logger.warning("Invalid tenant_id '%s'; falling back to 'default'.", tenant_id)
        tenant_id = "default"

    config_path = _TENANTS_DIR / f"{tenant_id}.yaml"
    if not config_path.exists():
        if tenant_id != "default":
            logger.warning("Tenant config not found: %s — using defaults.", config_path)
        return TenantContext(tenant_id=tenant_id)

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return TenantContext(
            tenant_id=tenant_id,
            display_name=data.get("display_name", tenant_id),
            prompt_versions=data.get("prompt_versions", {}),
            daily_budget_usd=float(data.get("daily_budget_usd", 50.0)),
            allowed_departments=data.get("allowed_departments", []),
            is_configured=True,
        )
    except ImportError:
        # yaml not available — return defaults
        logger.debug("PyYAML not available; using defaults for tenant '%s'.", tenant_id)
        return TenantContext(tenant_id=tenant_id)
    except Exception as exc:
        logger.error("Failed to load tenant config %s: %s — using defaults.", config_path, exc)
        return TenantContext(tenant_id=tenant_id)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_tenant_cache: dict = {}


def get_tenant(tenant_id: str = "default") -> TenantContext:
    """Cached version of get_tenant_context. Safe to call on every request."""
    if tenant_id not in _tenant_cache:
        _tenant_cache[tenant_id] = get_tenant_context(tenant_id)
    return _tenant_cache[tenant_id]


def clear_tenant_cache() -> None:
    """Clear the tenant context cache. Useful in tests."""
    _tenant_cache.clear()


def list_tenants() -> list:
    """Return sorted list of configured tenant IDs from the tenants/ directory."""
    if not _TENANTS_DIR.exists():
        return ["default"]
    return sorted(
        p.stem for p in _TENANTS_DIR.glob("*.yaml")
        if validate_tenant_id(p.stem)
    )
