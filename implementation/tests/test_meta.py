import inspect
from pathlib import Path

import puls_sched
from puls_sched.config import AdmissionConfig, ModelConfig, TimeConfig, default_dummy_config
from puls_sched.dag import DAG
from puls_sched.event import EventType
from puls_sched.main_loop import SchedulerCore
from puls_sched.node import NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request import RequestState


_EXPECTED_MODULES = {
    "__init__",
    "config",
    "clock",
    "request",
    "micro_batch",
    "node",
    "dag",
    "event",
    "event_queue",
    "window",
    "main_loop",
    "invariants",
    "dispatcher",
    # Impl-3
    "request_queue",
    "kv_accountant",
    "idle_telemetry",
    "deadband",
    "k_total",
    "admission",
    # Impl-4
    "pim_emulator",
}


_EXPECTED_K_TOTAL_DIAL = (0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048)

_EXPECTED_ADMISSION_FIELDS = {
    "n_sat",
    "kv_capacity_aggregate",
    "ctx_tier_short_max",
    "ctx_tier_mid_max",
    "deadband_width",
    "idle_theta_low",
    "idle_theta_high",
    "request_queue_capacity",
    "k_total_step",
    "k_total_max",
}


def test_meta_module_inventory():
    pkg_dir = Path(puls_sched.__file__).parent
    actual = {p.stem for p in pkg_dir.glob("*.py")}
    assert actual == _EXPECTED_MODULES, (
        f"module inventory mismatch — extra: {actual - _EXPECTED_MODULES}, "
        f"missing: {_EXPECTED_MODULES - actual}"
    )


def test_meta_node_types_complete():
    assert set(NodeType) == {
        NodeType.QKV,
        NodeType.PREFILL_ATTN,
        NodeType.DECODE_ATTN,
        NodeType.O_PROJ,
    }


def test_meta_request_states_complete():
    assert set(RequestState) == {
        RequestState.PENDING,
        RequestState.PREFILL,
        RequestState.DECODE,
        RequestState.COMPLETED,
    }


def test_meta_dag_precedence_matches_plan():
    """PLAN.md §6.3 의 I1·I2·I3 표와 정확 일치 (literal 비교)."""
    dag = DAG()
    dag.add_micro_batch(0)
    assert dag.precedence[0] == {
        NodeType.QKV: set(),
        NodeType.PREFILL_ATTN: {NodeType.QKV},                              # I1
        NodeType.DECODE_ATTN: {NodeType.QKV},                               # I2
        NodeType.O_PROJ: {NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN},     # I3
    }


def test_meta_k_total_dial_matches_plan_literal():
    """PLAN.md §4 Impl-3 의 k_total dial {0, 256, ..., 2048} (9-step) 정합."""
    cfg = default_dummy_config().admission
    dial = tuple(range(0, cfg.k_total_max + 1, cfg.k_total_step))
    assert dial == _EXPECTED_K_TOTAL_DIAL


def test_meta_admission_config_fields_match_plan_inventory():
    """AdmissionConfig 의 필드 set 이 PLAN.md §4 Impl-3 placeholder 목록 정합."""
    actual = set(AdmissionConfig.__dataclass_fields__.keys())
    assert actual == _EXPECTED_ADMISSION_FIELDS, (
        f"AdmissionConfig field mismatch — "
        f"extra: {actual - _EXPECTED_ADMISSION_FIELDS}, "
        f"missing: {_EXPECTED_ADMISSION_FIELDS - actual}"
    )


def test_meta_main_loop_handles_all_event_types():
    """SchedulerCore._handle 의 match case 가 EventType 전수 cover."""
    source = inspect.getsource(SchedulerCore._handle)
    for etype in EventType:
        assert f"EventType.{etype.name}" in source, (
            f"_handle missing case for {etype.name}"
        )


# =========================================================================
# Impl-4 — PLAN literal meta-test
# =========================================================================

def test_meta_model_config_has_kv_precision_field():
    """ModelConfig.kv_precision 필드 존재 + str 타입 (Q1, ARCH §3.1)."""
    fields = ModelConfig.__dataclass_fields__
    assert "kv_precision" in fields
    assert fields["kv_precision"].type is str


def test_meta_time_config_has_rtl_fsm_tile_rows_field():
    """TimeConfig.rtl_fsm_tile_rows 필드 존재 + int 타입 (ARCH §3.1 literal)."""
    fields = TimeConfig.__dataclass_fields__
    assert "rtl_fsm_tile_rows" in fields
    assert fields["rtl_fsm_tile_rows"].type is int


def test_meta_time_config_has_broadcast_latency_field():
    """TimeConfig.pim_broadcast_latency_ns_cross_gpu 필드 존재 + float 타입 (Q4)."""
    fields = TimeConfig.__dataclass_fields__
    assert "pim_broadcast_latency_ns_cross_gpu" in fields
    assert fields["pim_broadcast_latency_ns_cross_gpu"].type is float


def test_meta_pim_executor_method_inventory():
    """PIMExecutor 의 public method set bit-exact lock-in (Q7·Q8·Q9 정합)."""
    public_methods = {
        name for name, _ in inspect.getmembers(PIMExecutor, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_methods == {"tile_time", "op_time", "load_ramulator2_cycles"}


def test_meta_pim_tile_time_dict_has_both_regimes():
    """config.time.pim_tile_time_ns 가 FP8 + FP16 양 regime key 보유 (PLAN §3 literal)."""
    cfg = default_dummy_config()
    assert set(cfg.time.pim_tile_time_ns.keys()) == {"FP8", "FP16"}
