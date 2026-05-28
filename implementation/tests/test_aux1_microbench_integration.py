"""Cluster T4 — Aux1 T4 microbench JSON ingest integration.

PLAN §4 Impl-10 main + impl_10.md §5 Cluster T4. Colab T4 위 실측 산출의
empirical anchor — Aux1 closed-form 의 architectural property 정합 검증.

영역 정합 — T4 measured ratio = report markdown disclosure 만, 코드 산식 영역 ingest 0.
"""

import json
from pathlib import Path

import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.evaluator import Evaluator
from puls_sched.idle_telemetry import IdleTelemetry


T4_JSON_PATH = (
    Path(__file__).parent.parent / "data" / "aux1_ratio_t4_measured.json"
)


# ============================================================================
# T4.1 — JSON schema 정합 (metadata + batch_sweep_ms + aux1_ratio)
# ============================================================================


def test_t4_1_json_schema_top_level_keys():
    assert T4_JSON_PATH.exists(), f"Aux1 T4 JSON 부재: {T4_JSON_PATH}"
    data = json.loads(T4_JSON_PATH.read_text())
    assert "metadata" in data
    assert "batch_sweep_ms" in data
    assert "aux1_ratio" in data


def test_t4_1_json_metadata_provenance_label():
    data = json.loads(T4_JSON_PATH.read_text())
    meta = data["metadata"]
    assert meta["provenance_label"] == "colab_t4_free_llama3_70b_ffn_single_layer_measured"
    assert meta["gpu_name"].startswith("Tesla T4") or "T4" in meta["gpu_name"]
    assert meta["model_spec"]["hidden"] == 8192
    assert meta["model_spec"]["intermediate"] == 28672


# ============================================================================
# T4.2 — ratio 정성 영역 — memory-bound regime 위 > 1.0
# ============================================================================


def test_t4_2_aux1_ratio_memory_bound_regime_dominant():
    """Small batch (k ≤ 64) 위 t_separate / t_mixed > 1.5 — mixed batching 가속 확인."""
    data = json.loads(T4_JSON_PATH.read_text())
    ratios = data["aux1_ratio"]
    # k=8→16 + k=32→64 영역 — memory-bound dominant
    for key in ("8_to_16", "32_to_64"):
        if key in ratios:
            assert ratios[key]["ratio_separate_over_mixed"] > 1.5, \
                f"{key}: ratio={ratios[key]['ratio_separate_over_mixed']} expected >1.5 (memory-bound)"


# ============================================================================
# T4.3 — N_sat (saturating knee) 위치 노출 — batch sweep transition
# ============================================================================


def test_t4_3_n_sat_saturating_knee_exposed():
    """batch sweep 위 saturating knee 위치 노출 — compute-bound regime 위 ratio ≤ 1.2."""
    data = json.loads(T4_JSON_PATH.read_text())
    ratios = data["aux1_ratio"]
    # k=128→256 또는 k=512→1024 위 N_sat 진입 — ratio ≤ 1.2
    saturating_keys = ("128_to_256", "512_to_1024")
    found_saturating = False
    for key in saturating_keys:
        if key in ratios and "ratio_separate_over_mixed" in ratios[key]:
            r = ratios[key]["ratio_separate_over_mixed"]
            if r <= 1.2:
                found_saturating = True
                break
    assert found_saturating, "compute-bound regime (ratio ≤ 1.2) 영역 노출 안 됨"


# ============================================================================
# T4.4 — Evaluator.d2_report() 위 JSON ingest 정합
# ============================================================================


def test_t4_4_d2_report_ingests_aux1_t4_json():
    """d2_report 위 aux1_t4_ratio_path 위 JSON ingest + report 위 microbench section."""
    cfg = default_dummy_config()
    ev = Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())
    d2 = ev.d2_report(
        kv_lengths=[128000, 100000, 64000],
        aux1_t4_ratio_path=str(T4_JSON_PATH),
    )
    assert d2["aux1_t4_microbench"] is not None
    assert "metadata" in d2["aux1_t4_microbench"]
    assert "aux1_ratio" in d2["aux1_t4_microbench"]
    # markdown 위 T4 section disclosure
    assert "T4 Microbench" in d2["markdown"] or "Empirical Anchor" in d2["markdown"]
