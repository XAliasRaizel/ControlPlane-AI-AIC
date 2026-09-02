"""
backend/app/llm/prompt_registry.py

Versioned Jinja2 Prompt Registry for ControlPlane.ai.

Phase 3 Features:
  - Full semantic versioning: templates named v{major}.{minor}.{patch}.jinja2
  - Alias resolution via prompts/metadata.json:
      'production' -> 'v2.0.0', 'canary' -> 'v2.1.0', 'latest' -> 'v2.1.0'
  - SHA-256 prompt fingerprinting for audit reproducibility
  - Per-tenant version overrides via TenantContext
  - Backwards compat: legacy vN.jinja2 files still load correctly
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SEMVER_RE = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')
_SIMPLE_VER_RE = re.compile(r'^v(\d+)$')  # legacy vN format


def _parse_version_tuple(version_str: str) -> tuple:
    """Parse 'v2.1.0' -> (2, 1, 0). Also handles legacy 'v2' -> (2, 0, 0)."""
    m = _SEMVER_RE.match(version_str)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = _SIMPLE_VER_RE.match(version_str)
    if m:
        return int(m.group(1)), 0, 0
    raise ValueError(f"Cannot parse version string: '{version_str}'")


class PromptRegistry:
    """Load, version-resolve, and render Jinja2 prompt templates with SemVer support."""

    def __init__(self, prompts_dir=None) -> None:
        if prompts_dir is None:
            # 4 levels up from backend/app/llm/ -> project root -> prompts/
            prompts_dir = Path(__file__).resolve().parents[4] / "prompts"
        self._dir = Path(prompts_dir)
        self._aliases: dict = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load alias metadata from prompts/metadata.json if it exists."""
        meta_path = self._dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                # Strip internal _comment key
                self._aliases = {k: v for k, v in data.items() if not k.startswith("_")}
            except Exception as exc:
                logger.warning("Failed to load prompt metadata.json: %s", exc)

    def _resolve_version(self, prompt_name: str, version: str) -> str:
        """
        Resolve a version string:
          - 'latest', 'production', 'canary', 'stable' -> resolved from metadata.json
          - 'v2.1.0' -> returned as-is (exact SemVer)
          - 'v2'     -> returned as-is (legacy)
        """
        # Try alias resolution from metadata.json first
        if prompt_name in self._aliases and version in self._aliases[prompt_name]:
            resolved = self._aliases[prompt_name][version]
            logger.debug("Alias '%s' for '%s' -> '%s'", version, prompt_name, resolved)
            return resolved

        if version == "latest":
            return self.active_version(prompt_name)

        return version

    def _prompt_dir(self, prompt_name: str) -> Path:
        return self._dir / prompt_name

    def _template_path(self, prompt_name: str, version: str) -> Path:
        """Find template file for resolved version. Supports vN.M.P.jinja2 and vN.jinja2."""
        pdir = self._prompt_dir(prompt_name)
        # 1. Exact match (e.g. v2.1.0.jinja2 or v2.jinja2)
        exact = pdir / f"{version}.jinja2"
        if exact.exists():
            return exact

        # 2. SemVer fallback to legacy: v2.0.0 -> try v2.jinja2
        m = _SEMVER_RE.match(version)
        if m and m.group(2) == '0' and m.group(3) == '0':
            legacy = pdir / f"v{m.group(1)}.jinja2"
            if legacy.exists():
                return legacy

        # 3. Legacy fallback to SemVer: v2 -> try v2.0.0.jinja2
        m_simple = _SIMPLE_VER_RE.match(version)
        if m_simple:
            semver_equiv = pdir / f"v{m_simple.group(1)}.0.0.jinja2"
            if semver_equiv.exists():
                return semver_equiv

        raise FileNotFoundError(
            f"Prompt template '{prompt_name}/{version}.jinja2' not found in {pdir}"
        )

    def get(self, prompt_name: str, version: str = "latest") -> str:
        """Return the raw Jinja2 template string for a given prompt and version."""
        resolved = self._resolve_version(prompt_name, version)
        path = self._template_path(prompt_name, resolved)
        return path.read_text(encoding="utf-8")

    def render(self, prompt_name: str, version: str = "latest", **ctx) -> str:
        """Render a Jinja2 template with the given context variables."""
        try:
            import jinja2
            template_str = self.get(prompt_name, version)
            env = jinja2.Environment(undefined=jinja2.Undefined)
            tmpl = env.from_string(template_str)
            return tmpl.render(**ctx)
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to render prompt '%s' v%s: %s", prompt_name, version, exc)
            return self.get(prompt_name, version)

    def active_version(self, prompt_name: str) -> str:
        """
        Return the highest available version string for a prompt name.
        Considers both vN.M.P.jinja2 (SemVer) and legacy vN.jinja2 formats.
        """
        pdir = self._prompt_dir(prompt_name)
        if not pdir.exists():
            raise FileNotFoundError(f"Prompt directory '{prompt_name}' not found in {self._dir}")

        best = None
        best_str = ""
        for f in pdir.glob("*.jinja2"):
            stem = f.stem
            try:
                tup = _parse_version_tuple(stem)
                if best is None or tup > best:
                    best = tup
                    best_str = stem
            except ValueError:
                continue

        if not best_str:
            raise FileNotFoundError(f"No versioned templates found for '{prompt_name}'")
        return best_str

    def list_versions(self, prompt_name: str) -> list:
        """Return sorted list of all available version strings for a prompt."""
        pdir = self._prompt_dir(prompt_name)
        if not pdir.exists():
            return []
        versions = []
        for f in pdir.glob("*.jinja2"):
            try:
                _parse_version_tuple(f.stem)
                versions.append(f.stem)
            except ValueError:
                continue
        return sorted(versions, key=lambda v: _parse_version_tuple(v))

    def list_aliases(self, prompt_name: str) -> dict:
        """Return alias->version mapping for a prompt name."""
        return dict(self._aliases.get(prompt_name, {}))

    def prompt_fingerprint(self, prompt_name: str, version: str = "latest", **ctx) -> str:
        """Return SHA-256 hex digest of the rendered prompt (for audit immutability)."""
        rendered = self.render(prompt_name, version, **ctx)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


# Module-level singleton
_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
