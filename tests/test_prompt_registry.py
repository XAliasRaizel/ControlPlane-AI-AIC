"""
tests/test_prompt_registry.py

Tests for Phase 3 SemVer Prompt Registry:
- SemVer version string parsing
- Alias resolution (production/canary/stable/latest)
- Backwards compat with legacy vN.jinja2 files
- SHA-256 fingerprinting for audit
- Per-tenant version overrides
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSemVerParsing(unittest.TestCase):

    def test_semver_parse_full(self):
        from backend.app.llm.prompt_registry import _parse_version_tuple
        self.assertEqual(_parse_version_tuple("v2.1.0"), (2, 1, 0))
        self.assertEqual(_parse_version_tuple("v1.0.0"), (1, 0, 0))
        self.assertEqual(_parse_version_tuple("v10.5.3"), (10, 5, 3))

    def test_semver_parse_legacy(self):
        from backend.app.llm.prompt_registry import _parse_version_tuple
        self.assertEqual(_parse_version_tuple("v1"), (1, 0, 0))
        self.assertEqual(_parse_version_tuple("v2"), (2, 0, 0))

    def test_semver_parse_invalid_raises(self):
        from backend.app.llm.prompt_registry import _parse_version_tuple
        with self.assertRaises(ValueError):
            _parse_version_tuple("1.0.0")
        with self.assertRaises(ValueError):
            _parse_version_tuple("latest")


class TestPromptRegistryWithRealPrompts(unittest.TestCase):
    """Tests using the actual prompts/ directory."""

    def _make_registry(self):
        from backend.app.llm.prompt_registry import PromptRegistry
        root = Path(__file__).resolve().parents[1]
        return PromptRegistry(prompts_dir=root / "prompts")

    def test_get_ask_controlplane_v1_0_0(self):
        reg = self._make_registry()
        text = reg.get("ask_controlplane", version="v1.0.0")
        self.assertGreater(len(text), 20)
        self.assertIn("governance", text.lower())

    def test_get_ask_controlplane_v2_0_0(self):
        reg = self._make_registry()
        text = reg.get("ask_controlplane", version="v2.0.0")
        self.assertGreater(len(text), 20)

    def test_get_ask_controlplane_v2_1_0(self):
        reg = self._make_registry()
        text = reg.get("ask_controlplane", version="v2.1.0")
        self.assertIn("v2.1.0", text)

    def test_get_latest_returns_nonempty(self):
        reg = self._make_registry()
        text = reg.get("ask_controlplane", version="latest")
        self.assertGreater(len(text), 20)

    def test_list_versions_sorted_correctly(self):
        reg = self._make_registry()
        versions = reg.list_versions("ask_controlplane")
        self.assertIn("v1.0.0", versions)
        self.assertIn("v2.0.0", versions)
        idx1 = versions.index("v1.0.0")
        idx2 = versions.index("v2.0.0")
        self.assertLess(idx1, idx2)

    def test_active_version_is_highest(self):
        reg = self._make_registry()
        active = reg.active_version("ask_controlplane")
        versions = reg.list_versions("ask_controlplane")
        from backend.app.llm.prompt_registry import _parse_version_tuple
        active_tuple = _parse_version_tuple(active)
        for v in versions:
            self.assertLessEqual(_parse_version_tuple(v), active_tuple)

    def test_alias_production_resolves(self):
        reg = self._make_registry()
        aliases = reg.list_aliases("ask_controlplane")
        if aliases:
            text = reg.get("ask_controlplane", version="production")
            self.assertGreater(len(text), 20)

    def test_alias_canary_resolves(self):
        reg = self._make_registry()
        aliases = reg.list_aliases("ask_controlplane")
        if aliases:
            text = reg.get("ask_controlplane", version="canary")
            self.assertGreater(len(text), 20)

    def test_render_v2_0_0_with_department(self):
        reg = self._make_registry()
        text = reg.render("ask_controlplane", version="v2.0.0", department="Finance")
        self.assertIn("Finance", text)

    def test_render_v2_1_0_with_tenant_and_department(self):
        reg = self._make_registry()
        text = reg.render(
            "ask_controlplane", version="v2.1.0",
            department="finance", tenant_name="Acme Corp"
        )
        self.assertIn("Acme Corp", text)
        self.assertIn("FINANCE RULE", text)

    def test_render_v2_1_0_hr_department_rule(self):
        reg = self._make_registry()
        text = reg.render(
            "ask_controlplane", version="v2.1.0",
            department="hr", tenant_name="Test Corp"
        )
        self.assertIn("HR RULE", text)

    def test_prompt_fingerprint_is_64_char_hex(self):
        reg = self._make_registry()
        fp = reg.prompt_fingerprint("ask_controlplane", version="v1.0.0")
        self.assertEqual(len(fp), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in fp))

    def test_fingerprint_deterministic(self):
        reg = self._make_registry()
        fp1 = reg.prompt_fingerprint("ask_controlplane", version="v1.0.0")
        fp2 = reg.prompt_fingerprint("ask_controlplane", version="v1.0.0")
        self.assertEqual(fp1, fp2)

    def test_different_versions_have_different_fingerprints(self):
        reg = self._make_registry()
        fp1 = reg.prompt_fingerprint("ask_controlplane", version="v1.0.0")
        fp2 = reg.prompt_fingerprint("ask_controlplane", version="v2.0.0")
        self.assertNotEqual(fp1, fp2)

    def test_list_aliases_returns_dict(self):
        reg = self._make_registry()
        aliases = reg.list_aliases("ask_controlplane")
        self.assertIsInstance(aliases, dict)

    def test_rlhf_judge_v1_0_0_loads(self):
        reg = self._make_registry()
        text = reg.get("rlhf_judge", version="v1.0.0")
        self.assertIn("JSON", text)

    def test_grounding_extractor_v1_0_0_loads(self):
        reg = self._make_registry()
        text = reg.get("grounding_extractor", version="v1.0.0")
        self.assertIn("JSON", text)


class TestPromptRegistryIsolatedTempDir(unittest.TestCase):
    """Tests with a clean temp directory for full isolation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        prompts_dir = Path(self.tmpdir)
        (prompts_dir / "test_prompt").mkdir()
        (prompts_dir / "test_prompt" / "v1.0.0.jinja2").write_text(
            "Hello {{ name }}. Version 1.0.0.", encoding="utf-8"
        )
        (prompts_dir / "test_prompt" / "v2.0.0.jinja2").write_text(
            "Hello {{ name }}! Version 2.0.0 improved.", encoding="utf-8"
        )
        (prompts_dir / "test_prompt" / "v2.1.0.jinja2").write_text(
            "Hello {{ name }}! Version 2.1.0 canary.", encoding="utf-8"
        )
        meta = {
            "test_prompt": {
                "production": "v2.0.0",
                "canary": "v2.1.0",
                "stable": "v1.0.0",
                "latest": "v2.1.0"
            }
        }
        (prompts_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        from backend.app.llm.prompt_registry import PromptRegistry
        self.reg = PromptRegistry(prompts_dir=prompts_dir)

    def test_latest_resolves_via_alias(self):
        text = self.reg.get("test_prompt", version="latest")
        self.assertIn("2.1.0", text)

    def test_production_alias(self):
        text = self.reg.get("test_prompt", version="production")
        self.assertIn("2.0.0", text)

    def test_stable_alias(self):
        text = self.reg.get("test_prompt", version="stable")
        self.assertIn("1.0.0", text)

    def test_canary_alias(self):
        text = self.reg.get("test_prompt", version="canary")
        self.assertIn("2.1.0", text)

    def test_render_with_context_variable(self):
        text = self.reg.render("test_prompt", version="v1.0.0", name="Ayush")
        self.assertIn("Ayush", text)

    def test_active_version_is_v2_1_0(self):
        self.assertEqual(self.reg.active_version("test_prompt"), "v2.1.0")

    def test_list_versions_ordered(self):
        versions = self.reg.list_versions("test_prompt")
        self.assertEqual(versions, ["v1.0.0", "v2.0.0", "v2.1.0"])

    def test_fingerprint_is_deterministic_64_chars(self):
        fp1 = self.reg.prompt_fingerprint("test_prompt", version="v1.0.0", name="x")
        fp2 = self.reg.prompt_fingerprint("test_prompt", version="v1.0.0", name="x")
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_list_aliases_returns_correct_dict(self):
        aliases = self.reg.list_aliases("test_prompt")
        self.assertEqual(aliases["production"], "v2.0.0")
        self.assertEqual(aliases["canary"], "v2.1.0")
        self.assertEqual(aliases["stable"], "v1.0.0")


if __name__ == "__main__":
    unittest.main()
