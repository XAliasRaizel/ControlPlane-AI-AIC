"""
tests/test_multi_tenant.py

Tests for Phase 3 multi-tenant namespacing:
- TenantContext loads from YAML configs correctly
- VectorStore namespaces collection names correctly
- Data isolation between tenants (zero cross-tenant leakage)
- tenant_id validation rejects dangerous strings
- RagSettings has multi-tenant fields
"""
from __future__ import annotations

import sys
import tempfile
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestTenantContext(unittest.TestCase):

    def setUp(self):
        from rag.tenant import clear_tenant_cache
        clear_tenant_cache()

    def test_default_tenant_has_no_namespace(self):
        from rag.tenant import TenantContext
        ctx = TenantContext(tenant_id="default")
        self.assertEqual(ctx.namespace, "")
        self.assertEqual(ctx.namespaced_collection("policy_evidence"), "policy_evidence")

    def test_non_default_tenant_namespaces_collection(self):
        from rag.tenant import TenantContext
        ctx = TenantContext(tenant_id="acme_corp")
        self.assertEqual(ctx.namespace, "acme_corp__")
        self.assertEqual(ctx.namespaced_collection("policy_evidence"), "acme_corp__policy_evidence")

    def test_validate_tenant_id_allows_valid_ids(self):
        from rag.tenant import validate_tenant_id
        self.assertTrue(validate_tenant_id("default"))
        self.assertTrue(validate_tenant_id("acme_corp"))
        self.assertTrue(validate_tenant_id("tenant-123"))
        self.assertTrue(validate_tenant_id("my.tenant"))

    def test_validate_tenant_id_rejects_bad_ids(self):
        from rag.tenant import validate_tenant_id
        self.assertFalse(validate_tenant_id(""))
        self.assertFalse(validate_tenant_id("a" * 65))
        self.assertFalse(validate_tenant_id("tenant/hack"))
        self.assertFalse(validate_tenant_id("tenant hack"))
        self.assertFalse(validate_tenant_id("tenant#evil"))

    def test_get_tenant_returns_defaults_for_missing_config(self):
        from rag.tenant import get_tenant
        ctx = get_tenant("nonexistent_tenant_xyz_123")
        self.assertEqual(ctx.tenant_id, "nonexistent_tenant_xyz_123")
        self.assertFalse(ctx.is_configured)

    def test_tenant_prompt_version_default(self):
        from rag.tenant import TenantContext
        ctx = TenantContext(tenant_id="default")
        self.assertEqual(ctx.prompt_version_for("ask_controlplane"), "latest")

    def test_tenant_prompt_version_override(self):
        from rag.tenant import TenantContext
        ctx = TenantContext(
            tenant_id="acme",
            prompt_versions={"ask_controlplane": "production"}
        )
        self.assertEqual(ctx.prompt_version_for("ask_controlplane"), "production")
        self.assertEqual(ctx.prompt_version_for("rlhf_judge"), "latest")  # default fallback

    def test_list_tenants_includes_default(self):
        from rag.tenant import list_tenants
        tenants = list_tenants()
        self.assertIn("default", tenants)

    def test_list_tenants_includes_acme_corp(self):
        from rag.tenant import list_tenants
        tenants = list_tenants()
        self.assertIn("acme_corp", tenants)

    def test_acme_corp_tenant_loads(self):
        from rag.tenant import get_tenant
        ctx = get_tenant("acme_corp")
        self.assertEqual(ctx.tenant_id, "acme_corp")
        # Should be configured if yaml exists
        # (is_configured=True only when yaml file exists)

    def test_demo_tenant_loads(self):
        from rag.tenant import get_tenant
        ctx = get_tenant("demo_tenant")
        self.assertEqual(ctx.tenant_id, "demo_tenant")


class TestVectorStoreNamespacing(unittest.TestCase):

    def test_default_tenant_does_not_modify_collection_name(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore("policy_evidence", persist_dir=tmpdir, tenant_id="default")
            self.assertEqual(store.collection_name, "policy_evidence")

    def test_tenant_namespaces_collection_name(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore("policy_evidence", persist_dir=tmpdir, tenant_id="acme_corp")
            self.assertEqual(store.collection_name, "acme_corp__policy_evidence")

    def test_two_tenants_use_different_stores_no_leakage(self):
        """Upsert to tenant A must not appear when querying tenant B's store."""
        import numpy as np
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store_a = VectorStore("test_col", persist_dir=tmpdir, tenant_id="tenant_alpha")
            store_b = VectorStore("test_col", persist_dir=tmpdir, tenant_id="tenant_beta")

            rng = np.random.default_rng(42)
            emb = list(rng.random(128).astype(float))
            store_a.upsert(
                ids=["secret_doc"],
                texts=["Tenant Alpha confidential salary data"],
                embeddings=[emb],
                metadatas=[{"source": "alpha_hr"}],
            )

            # Tenant Beta must see zero results — strict isolation
            results = store_b.query(emb, top_k=5)
            self.assertEqual(len(results), 0,
                             f"Tenant Beta leaked data from Tenant Alpha! Got {results}")

    def test_upsert_injects_tenant_id_into_metadata(self):
        """Documents upserted for a non-default tenant carry tenant_id in their metadata."""
        import numpy as np
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore("test_col", persist_dir=tmpdir, tenant_id="acme_corp")
            rng = np.random.default_rng(7)
            emb = list(rng.random(128).astype(float))
            store.upsert(
                ids=["doc1"],
                texts=["acme policy content"],
                embeddings=[emb],
                metadatas=[{"source": "hr_policy"}],
            )
            results = store.query(emb, top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"].get("tenant_id"), "acme_corp")

    def test_default_tenant_does_not_inject_tenant_id(self):
        """Default tenant metadata should NOT have tenant_id injected."""
        import numpy as np
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore("test_col", persist_dir=tmpdir, tenant_id="default")
            rng = np.random.default_rng(99)
            emb = list(rng.random(128).astype(float))
            store.upsert(
                ids=["doc1"],
                texts=["default policy content"],
                embeddings=[emb],
                metadatas=[{"source": "policy"}],
            )
            results = store.query(emb, top_k=1)
            self.assertEqual(len(results), 1)
            # tenant_id should NOT be injected for default tenant
            self.assertNotIn("tenant_id", results[0]["metadata"])


class TestRagConfigMultiTenant(unittest.TestCase):

    def test_config_has_multi_tenant_enabled_field(self):
        from rag.config import RagSettings
        settings = RagSettings()
        self.assertIsInstance(settings.multi_tenant_enabled, bool)

    def test_config_has_default_tenant_id_field(self):
        from rag.config import RagSettings
        settings = RagSettings()
        self.assertIsInstance(settings.default_tenant_id, str)
        self.assertEqual(settings.default_tenant_id, "default")

    def test_multi_tenant_disabled_by_default(self):
        from unittest.mock import patch
        from rag.config import RagSettings
        env = os.environ.copy()
        env.pop("RAG_MULTI_TENANT_ENABLED", None)
        with patch.dict(os.environ, env, clear=True):
            settings = RagSettings()
            self.assertFalse(settings.multi_tenant_enabled)

    def test_multi_tenant_enabled_via_env(self):
        from unittest.mock import patch
        from rag.config import RagSettings
        with patch.dict(os.environ, {"RAG_MULTI_TENANT_ENABLED": "true"}):
            settings = RagSettings()
            self.assertTrue(settings.multi_tenant_enabled)


if __name__ == "__main__":
    unittest.main()
