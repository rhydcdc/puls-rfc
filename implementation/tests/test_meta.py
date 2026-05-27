import inspect
from pathlib import Path

import puls_sched
from puls_sched.admission import MicroBatchSpec
from puls_sched.config import AdmissionConfig, ModelConfig, TimeConfig, default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import EventType
from puls_sched.forward_pass import ForwardPass, LayerState
from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.main_loop import SchedulerCore
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType
from puls_sched.nvlink import NVLinkTransfer
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
    # Impl-5
    "instance",
    "nvlink",
    "instance_pipeline",
    "forward_pass",
    # Impl-6
    "trace",
    "completion",
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


# =========================================================================
# Impl-5 — PLAN literal meta-test
# =========================================================================

def test_meta_micro_batch_has_impl5_fields():
    """MicroBatch 의 Impl-5 신규 3 필드 존재 + int."""
    fields = MicroBatch.__dataclass_fields__
    for name in ("k_total", "kv_rows_total", "current_layer_index"):
        assert name in fields
        assert fields[name].type is int


def test_meta_micro_batch_spec_has_kv_rows_total_field():
    fields = MicroBatchSpec.__dataclass_fields__
    assert "kv_rows_total" in fields
    assert fields["kv_rows_total"].type is int


def test_meta_dispatcher_has_micro_batches_field():
    assert "micro_batches" in Dispatcher.__dataclass_fields__


def test_meta_dispatcher_has_register_method():
    assert hasattr(Dispatcher, "register")
    sig = inspect.signature(Dispatcher.register)
    assert list(sig.parameters.keys()) == ["self", "mb"]


def test_meta_instance_class_fields():
    assert set(Instance.__dataclass_fields__.keys()) == {"name", "has_pim", "gpu_busy", "pim_busy"}


def test_meta_instance_pipeline_class_fields():
    assert set(InstancePipeline.__dataclass_fields__.keys()) == {
        "config", "instance_a", "instance_b", "nvlink",
    }


def test_meta_nvlink_transfer_class_fields():
    assert set(NVLinkTransfer.__dataclass_fields__.keys()) == {"config", "bytes_per_element"}


def test_meta_forward_pass_class_fields():
    assert set(ForwardPass.__dataclass_fields__.keys()) == {
        "config", "instance_pipeline", "layer_state",
    }


def test_meta_layer_state_class_fields():
    assert set(LayerState.__dataclass_fields__.keys()) == {"num_layers"}


def test_meta_arch_3_4_case_a_gpus_total():
    """ARCH §3.4 Case A literal — Instance A 8 + Instance B 8 = 16 GPUs total."""
    cfg = default_dummy_config()
    assert cfg.hw.num_gpus_instance_a + cfg.hw.num_gpus_instance_b == 16


def test_meta_arch_3_4_inter_instance_data_decode_shape():
    """ARCH §3.4 표 — A → B: O projection output [B × hidden]. NVLinkTransfer.time signature 정합."""
    sig = inspect.signature(NVLinkTransfer.time)
    assert "tensor_shape" in sig.parameters


# ============================================================================
# Impl-6 meta — Q lock-in + ARCH §3.3 / §8 literal + data directory
# ============================================================================

def test_meta_request_has_impl6_lifecycle_fields():
    """Q10 (b) — Request 가 lifecycle owner (max_tokens, decoded_count, completion_time)"""
    from puls_sched.request import Request
    fields = Request.__dataclass_fields__
    assert "max_tokens" in fields
    assert "decoded_count" in fields
    assert "completion_time" in fields
    assert fields["max_tokens"].type is int
    assert fields["decoded_count"].type is int


def test_meta_trace_entry_fields():
    from puls_sched.trace import TraceEntry
    assert set(TraceEntry.__dataclass_fields__.keys()) == {
        "arrived_at", "num_prefill_tokens", "num_decode_tokens"
    }


def test_meta_trace_replayer_methods():
    from puls_sched.trace import TraceReplayer
    publics = {n for n in dir(TraceReplayer)
               if not n.startswith("_") and callable(getattr(TraceReplayer, n))}
    # entries 는 field — load · replay · stats 만 public method
    assert {"load", "replay", "stats"}.issubset(publics)


def test_meta_trace_supported_schemas_longbench_only():
    """Q2 lock-in — longbench_csv 만 지원"""
    from puls_sched.trace import _SUPPORTED_SCHEMAS
    assert _SUPPORTED_SCHEMAS == ("longbench_csv",)


def test_meta_trace_phase_3_schemas_inventory():
    """Q8 lock-in — 1M-class + mid-ctx 가 Phase 3 stub"""
    from puls_sched.trace import _PHASE_3_SCHEMAS
    assert _PHASE_3_SCHEMAS == ("vidur_1m_class", "mooncake_chat")


def test_meta_completion_methods():
    from puls_sched.completion import Completion
    publics = {n for n in dir(Completion)
               if not n.startswith("_") and callable(getattr(Completion, n))}
    assert {"check", "finalize"}.issubset(publics)


def test_meta_completion_class_fields():
    """Q9 책임 분리 — Completion 에 dispatcher 없음"""
    from puls_sched.completion import Completion
    assert set(Completion.__dataclass_fields__.keys()) == {"clock", "kv_accountant"}


def test_meta_scheduler_core_has_layer_state_field():
    """Q5 · Q10 lifecycle owner 패턴"""
    fields = SchedulerCore.__dataclass_fields__
    assert "layer_state" in fields
    assert "completion" in fields
    assert "in_flight_requests" in fields


def test_meta_arch_3_3_kv_resident_invariant():
    """ARCH §3.3 — KV admit ↔ release round-trip 의 보존성"""
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request import Request
    kv = KVAccountant(capacity=1000)
    req = Request(id=0, prompt_tokens=[], kv_length=100)
    kv.admit(req)
    assert kv.remaining == 900
    kv.release(req)
    assert kv.remaining == 1000


def test_meta_arch_8_trace_area_longctx_only():
    """ARCH §8 — Impl-6 의 trace 영역은 long-ctx 만 (1M / mid-ctx 부재)"""
    from puls_sched.trace import _SUPPORTED_SCHEMAS, _PHASE_3_SCHEMAS
    assert "vidur_1m_class" not in _SUPPORTED_SCHEMAS
    assert "mooncake_chat" not in _SUPPORTED_SCHEMAS


def test_meta_data_directory_traces_present():
    """Q1 lock-in — implementation/data/ 의 두 trace 파일 존재 + 비어있지 않음"""
    data_dir = Path(__file__).parent.parent / "data"
    p1 = data_dir / "longctx_longbench_lambda_3_40.csv"
    p2 = data_dir / "longctx_longbench_lambda_6_67.csv"
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0
