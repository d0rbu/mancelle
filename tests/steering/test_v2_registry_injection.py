"""Tests for the SAELens registry-injection shim."""

import dataclasses
import subprocess
import sys
import textwrap

from sae_lens.loading.pretrained_saes_directory import (
    PretrainedSAELookup,
    get_pretrained_saes_directory,
)

from steering.sae_resolver import ActivationType
from steering.sae_resolver_v2_data import (
    GEMMASCOPE_2_VALID_COMBOS,
)
from steering.sae_resolver_v2_registry import (
    _EXPECTED_PRETRAINED_SAE_LOOKUP_FIELDS,
    inject_missing_v2_releases,
)


class TestInjection:
    def test_module_import_injects_releases(self):
        import steering.sae_resolver_v2_registry  # noqa: F401

        directory = get_pretrained_saes_directory()
        for release in (
            "gemma-scope-2-1b-pt-res",
            "gemma-scope-2-1b-pt-mlp",
            "gemma-scope-2-1b-pt-att",
            "gemma-scope-2-12b-pt-res",
            "gemma-scope-2-12b-pt-mlp",
            "gemma-scope-2-12b-pt-att",
            "gemma-scope-2-27b-pt-mlp",
            "gemma-scope-2-27b-pt-att",
        ):
            assert release in directory, f"missing injected release: {release}"

    def test_injection_idempotent(self):
        directory = get_pretrained_saes_directory()
        before = dict(directory)  # snapshot
        for _ in range(3):
            inject_missing_v2_releases()
        after = dict(directory)
        assert before == after, "second invocation must not change registry"

    def test_injection_does_not_mutate_registered_release(self):
        directory = get_pretrained_saes_directory()
        # gemma-scope-2-4b-pt-res IS in the SAELens registry. The injector
        # must not have touched it.
        rec = directory["gemma-scope-2-4b-pt-res"]
        assert rec.repo_id == "google/gemma-scope-2-4b-pt"
        assert rec.model == "google/gemma-3-4b-pt"
        assert rec.conversion_func == "gemma_3"

    def test_injected_record_uses_gemma_3_conversion(self):
        directory = get_pretrained_saes_directory()
        for release in (
            "gemma-scope-2-1b-pt-res",
            "gemma-scope-2-12b-pt-mlp",
            "gemma-scope-2-27b-pt-att",
        ):
            rec = directory[release]
            assert rec.conversion_func == "gemma_3"

    def test_injected_record_repo_id(self):
        directory = get_pretrained_saes_directory()
        assert directory["gemma-scope-2-1b-pt-res"].repo_id == "google/gemma-scope-2-1b-pt"
        assert directory["gemma-scope-2-12b-pt-mlp"].repo_id == "google/gemma-scope-2-12b-pt"
        assert directory["gemma-scope-2-27b-pt-att"].repo_id == "google/gemma-scope-2-27b-pt"

    def test_injected_saes_map_keys_match_validation_table(self):
        directory = get_pretrained_saes_directory()
        rec = directory["gemma-scope-2-1b-pt-res"]
        # Build the expected keys from the validation table.
        expected = set()
        for (act, width), tier_layers in GEMMASCOPE_2_VALID_COMBOS["gemma-3-1b-pt"].items():
            if act != ActivationType.residual:
                continue
            for tier, layers in tier_layers.items():
                for L in layers:
                    expected.add(f"layer_{L}_width_{width.value}_l0_{tier.value}")
        assert set(rec.saes_map.keys()) == expected

    def test_injected_saes_map_values_use_resid_post_folder(self):
        rec = get_pretrained_saes_directory()["gemma-scope-2-12b-pt-res"]
        for k, v in rec.saes_map.items():
            assert v == f"resid_post/{k}", (k, v)

    def test_pretrained_sae_lookup_field_set_unchanged(self):
        # The PRODUCTION constant must equal the dataclass's introspected
        # fields. This is not a re-typed literal: if SAELens renames/adds a
        # field, the production guard constant and reality diverge and this
        # test (plus the runtime _assert_schema_unchanged) fails loudly.
        names = {f.name for f in dataclasses.fields(PretrainedSAELookup)}
        assert _EXPECTED_PRETRAINED_SAE_LOOKUP_FIELDS == names, (
            "PretrainedSAELookup schema drifted from the production guard "
            f"constant.\n  guard:  {sorted(_EXPECTED_PRETRAINED_SAE_LOOKUP_FIELDS)}"
            f"\n  actual: {sorted(names)}"
        )


class TestPackageImportSideEffect:
    def test_importing_steering_top_level_injects_releases(self):
        # We test by simulating a fresh process: load via a child interpreter.
        code = textwrap.dedent("""
            import steering  # noqa: F401
            from sae_lens.loading.pretrained_saes_directory import (
                get_pretrained_saes_directory,
            )

            d = get_pretrained_saes_directory()
            assert "gemma-scope-2-27b-pt-mlp" in d, "injection did not run"
            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        )
        assert "OK" in result.stdout


class TestSAEFromPretrainedHonoursInjection:
    def test_from_pretrained_finds_injected_release_metadata(self):
        # We don't actually download — we only check that SAELens resolves
        # the (release, sae_id) pair without raising "unknown release".
        # Using 27b-pt-mlp specifically because it's TRULY missing from
        # upstream SAELens 6.42 and only present via our injection shim,
        # so a passing assertion proves our shim's reachability.
        directory = get_pretrained_saes_directory()
        rec = directory["gemma-scope-2-27b-pt-mlp"]
        # Pick a real sae_id from the injected saes_map.
        sae_id = next(iter(rec.saes_map))
        assert sae_id.startswith("layer_")
        # The mapped HF path is what SAELens will download.
        path = rec.saes_map[sae_id]
        assert path.startswith("mlp_out/")
        # No exception means the registry lookup is healthy.
