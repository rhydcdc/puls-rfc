"""ARCH signature divergence lock-in.

PLAN.md §4 Impl-4 의 초기 signature `op_time(k_channels, N_decode, N_kv_avg)` 가
ARCH 의 정확 반영 후 `(k_channels, kv_rows_total)` 로 갱신된 *의도* 를 영구 기록.
누군가 향후 N_decode arg / dispatch method / clock·queue field 를 *재추가* 하면 자동 fail.

ARCH 근거:
- §3.1 "FSM cycle structure is invariant whether GEMV (B=1) or GEMM (B>1)"
  → batch dim arg 부재
- §3.1 "FP16 MAC core / FP8 (E4M3) KV-cache storage"
  → regime = system-wide → regime arg 부재
- §3.5.2 "no separate synchronization mechanism required — completion notification
  · interrupt · barrier all unnecessary"
  → dispatch method 부재
- §3.4 "KV rows are sharded across channels"
  → kv_rows_total (exact sum, not average) arg
"""

import inspect

from puls_sched.dispatcher import Dispatcher
from puls_sched.forward_pass import ForwardPass, LayerState
from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.nvlink import NVLinkTransfer
from puls_sched.pim_emulator import PIMExecutor


def test_op_time_signature_matches_arch_3_1_invariance():
    """op_time signature = {self, k_channels, kv_rows_total, kv_rows_lockstep}. N_decode 부재 (ARCH §3.1).
    Impl-8 — kv_rows_lockstep 추가 (F5 ablation 위, default 0 backward-compat)."""
    params = set(inspect.signature(PIMExecutor.op_time).parameters.keys())
    assert params == {"self", "k_channels", "kv_rows_total", "kv_rows_lockstep"}, (
        f"PIMExecutor.op_time signature divergence — {params}. "
        f"ARCH §3.1 'FSM cycle structure invariant' 정합 으로 batch arg 부재 요구."
    )


def test_op_time_no_regime_arg():
    """op_time signature 에 regime / kv_precision arg 부재 (Q1·Q9, ARCH §3.1)."""
    params = set(inspect.signature(PIMExecutor.op_time).parameters.keys())
    assert "regime" not in params
    assert "kv_precision" not in params


def test_tile_time_no_regime_arg():
    """tile_time signature 도 self 외 0 arg (Q9, system-wide config lookup)."""
    params = set(inspect.signature(PIMExecutor.tile_time).parameters.keys())
    assert params == {"self"}, (
        f"PIMExecutor.tile_time signature divergence — {params}. "
        f"regime 은 self.config.model.kv_precision lookup."
    )


def test_pim_executor_no_dispatch_method():
    """PIMExecutor 에 dispatch method 부재 (Q7, ARCH §3.5.2)."""
    assert not hasattr(PIMExecutor, "dispatch"), (
        "PIMExecutor 가 dispatch method 보유 — ARCH §3.5.2 'no separate "
        "synchronization mechanism required' 와 불일치. PIMExecutor 는 "
        "시간 계산기, dispatch agent 아님."
    )


def test_pim_executor_no_clock_or_queue_field():
    """PIMExecutor dataclass field set = {config} only (Q8, stateless)."""
    fields = set(PIMExecutor.__dataclass_fields__.keys())
    assert fields == {"config"}, (
        f"PIMExecutor fields divergence — {fields}. "
        f"clock · queue 필드 추가 시 stateless 정합 깨짐 (ARCH §3.5.2)."
    )


# =========================================================================
# Impl-5 signature divergence lock-in
# =========================================================================

def test_instance_pipeline_no_l_loop():
    """InstancePipeline 에 L-loop method 부재 (Q3 — forward_pass = L-loop owner)."""
    for forbidden in ("run", "iterate_layers", "forward"):
        assert not hasattr(InstancePipeline, forbidden), (
            f"InstancePipeline.{forbidden} 존재 — Q3 위반 (L-loop 은 ForwardPass 책임)."
        )


def test_forward_pass_owns_l_loop():
    """ForwardPass.run 존재 + signature `(self, mb)`."""
    assert hasattr(ForwardPass, "run")
    params = list(inspect.signature(ForwardPass.run).parameters.keys())
    assert params == ["self", "mb"]


def test_nvlink_no_event_push():
    """NVLinkTransfer public method = {time} only (Q4 — pure function, event push 부재)."""
    public_methods = {
        name for name, _ in inspect.getmembers(NVLinkTransfer, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_methods == {"time"}, (
        f"NVLinkTransfer method divergence — {public_methods}. "
        f"Q4 정합 — pure time function, event push 안 함."
    )


def test_nvlink_no_clock_or_queue_field():
    """NVLinkTransfer field set = {config, bytes_per_element} only (Q4 stateless)."""
    fields = set(NVLinkTransfer.__dataclass_fields__.keys())
    assert fields == {"config", "bytes_per_element"}, (
        f"NVLinkTransfer fields divergence — {fields}."
    )


def test_instance_pipeline_steady_state_runtime_getter():
    """InstancePipeline.steady_state_cycle 존재 + signature `(self, a_cycle, b_cycle)` (Q6)."""
    assert hasattr(InstancePipeline, "steady_state_cycle")
    params = list(inspect.signature(InstancePipeline.steady_state_cycle).parameters.keys())
    assert params == ["self", "a_cycle", "b_cycle"]


def test_dispatcher_register_signature():
    """Dispatcher.register signature `(self, mb)` (Q1-bis)."""
    params = list(inspect.signature(Dispatcher.register).parameters.keys())
    assert params == ["self", "mb"]


def test_instance_has_pim_field():
    """Instance.has_pim 필드 + bool 타입 (Q2 — Instance A/B 구조 차이)."""
    fields = Instance.__dataclass_fields__
    assert "has_pim" in fields
    assert fields["has_pim"].type is bool


def test_layer_state_advance_signature():
    """LayerState.advance signature `(self, mb)` → bool (token decode signal)."""
    params = list(inspect.signature(LayerState.advance).parameters.keys())
    assert params == ["self", "mb"]


# =========================================================================
# Impl-6 signature divergence lock-in
# =========================================================================

def test_trace_replayer_no_rng_field():
    """Q4 — TraceReplayer 에 RNG / seed / random_state field 부재 (determinism 자연 보존)"""
    from puls_sched.trace import TraceReplayer
    fields = set(TraceReplayer.__dataclass_fields__.keys())
    forbidden = {"rng", "seed", "random_state", "rand"}
    assert not (fields & forbidden), (
        f"TraceReplayer 가 RNG field 보유: {fields & forbidden}. "
        f"Q4 정합 — pure stateless, RNG 의존 0."
    )


def test_trace_replayer_load_static_method():
    """load → instance 패턴 — staticmethod 로 노출"""
    from puls_sched.trace import TraceReplayer
    # __dict__ 접근으로 staticmethod 확인 (dataclass 가 wrap 안 함)
    assert isinstance(TraceReplayer.__dict__["load"], staticmethod)


def test_trace_replay_signature():
    """TraceReplayer.replay signature `(self, rate_multiplier=1.0)`"""
    from puls_sched.trace import TraceReplayer
    params = list(inspect.signature(TraceReplayer.replay).parameters.keys())
    assert params == ["self", "rate_multiplier"]


def test_completion_check_signature():
    """Completion.check signature `(self, req, eos_seen=False)` — Q6 EOS branch lock-in"""
    from puls_sched.completion import Completion
    sig = inspect.signature(Completion.check)
    params = list(sig.parameters.keys())
    assert params == ["self", "req", "eos_seen"]
    assert sig.parameters["eos_seen"].default is False


def test_completion_finalize_signature():
    """Completion.finalize signature `(self, req)` — dispatcher arg 부재 (Q9)"""
    from puls_sched.completion import Completion
    params = list(inspect.signature(Completion.finalize).parameters.keys())
    assert params == ["self", "req"]
    assert "dispatcher" not in params
    assert "mb" not in params


def test_completion_no_dispatcher_field():
    """Q9 책임 분리 — Completion field 에 dispatcher 부재"""
    from puls_sched.completion import Completion
    fields = set(Completion.__dataclass_fields__.keys())
    forbidden = {"dispatcher", "micro_batch", "mb"}
    assert not (fields & forbidden), (
        f"Completion 이 dispatcher field 보유: {fields & forbidden}. "
        f"Q9 책임 분리 위반."
    )


def test_request_lifecycle_owner_pattern():
    """Q10 (b) — Request 가 decoded_count + max_tokens + completion_time 보유"""
    from puls_sched.request import Request
    fields = Request.__dataclass_fields__
    assert "decoded_count" in fields
    assert "max_tokens" in fields
    assert "completion_time" in fields


def test_main_loop_token_decode_signal_method():
    """Q5 — SchedulerCore._maybe_advance_forward_pass 존재"""
    from puls_sched.main_loop import SchedulerCore
    assert hasattr(SchedulerCore, "_maybe_advance_forward_pass")


def test_main_loop_in_flight_requests_dict():
    """Q10 lifecycle owner — SchedulerCore.in_flight_requests field 존재"""
    from puls_sched.main_loop import SchedulerCore
    fields = SchedulerCore.__dataclass_fields__
    assert "in_flight_requests" in fields


# ============================================================================
# Impl-8 — Evaluator + ablation wiring signature lock-in
# ============================================================================

def test_evaluator_record_dispatch_signature():
    """Evaluator.record_dispatch signature == (self, event: DispatchEvent) -> None"""
    from puls_sched.evaluator import Evaluator
    sig = inspect.signature(Evaluator.record_dispatch)
    assert list(sig.parameters.keys()) == ["self", "event"]


def test_evaluator_record_admission_tick_signature():
    """Evaluator.record_admission_tick signature == (self, snapshot)"""
    from puls_sched.evaluator import Evaluator
    sig = inspect.signature(Evaluator.record_admission_tick)
    assert list(sig.parameters.keys()) == ["self", "snapshot"]


def test_evaluator_no_baseline_method():
    """Comparative baseline 부재 lock-in — method name 에 baseline/sarathi/vllm/compare 부재."""
    from puls_sched.evaluator import Evaluator
    forbidden = {"baseline", "sarathi", "vllm", "compare"}
    for name in dir(Evaluator):
        for f in forbidden:
            assert f not in name.lower(), f"Evaluator.{name} contains forbidden term '{f}'"


def test_evaluator_no_absolute_metric_method():
    """절대 metric 부재 lock-in — method name 에 ttft/tpot/throughput/goodput/slo 부재."""
    from puls_sched.evaluator import Evaluator
    forbidden = {"ttft", "tpot", "throughput", "goodput", "slo"}
    for name in dir(Evaluator):
        for f in forbidden:
            assert f not in name.lower(), f"Evaluator.{name} contains forbidden term '{f}'"


def test_dispatcher_on_dispatch_signature():
    """Dispatcher.on_dispatch == (self, callback) -> None (D1 hook 등록 API)"""
    sig = inspect.signature(Dispatcher.on_dispatch)
    assert list(sig.parameters.keys()) == ["self", "callback"]


def test_dispatcher_dispatch_callbacks_field():
    """Dispatcher._dispatch_callbacks field 존재 (D1 hook 영구 기록)"""
    assert "_dispatch_callbacks" in Dispatcher.__dataclass_fields__


def test_scheduler_core_on_admission_tick_signature():
    """SchedulerCore.on_admission_tick == (self, callback)"""
    from puls_sched.main_loop import SchedulerCore
    sig = inspect.signature(SchedulerCore.on_admission_tick)
    assert list(sig.parameters.keys()) == ["self", "callback"]


def test_window_no_class_const_capacity():
    """InFlightWindow.CAPACITY 부재 (rename → DEFAULT_CAPACITY + instance.capacity)."""
    from puls_sched.window import InFlightWindow
    assert not hasattr(InFlightWindow, "CAPACITY"), (
        "InFlightWindow.CAPACITY 가 instance field 화되어야 함 (F2 ablation 위)"
    )
    assert hasattr(InFlightWindow, "DEFAULT_CAPACITY")


def test_ablation_config_is_frozen():
    """AblationConfig frozen dataclass — immutable lock-in."""
    import dataclasses as dc
    from puls_sched.config import AblationConfig
    assert AblationConfig.__dataclass_params__.frozen is True


def test_evaluator_is_dataclass():
    """Evaluator 는 dataclass — field 기반 inspector 가능."""
    import dataclasses as dc
    from puls_sched.evaluator import Evaluator
    assert dc.is_dataclass(Evaluator)


def test_pim_emulator_op_time_signature_has_kv_rows_lockstep():
    """PIMExecutor.op_time signature 에 kv_rows_lockstep param + default 0 (Q8 backward-compat)."""
    sig = inspect.signature(PIMExecutor.op_time)
    assert "kv_rows_lockstep" in sig.parameters
    assert sig.parameters["kv_rows_lockstep"].default == 0


def test_evaluator_no_dispatcher_field():
    """D3 — Evaluator 가 dispatcher/scheduler_core reference 미보유 (standalone)."""
    from puls_sched.evaluator import Evaluator
    fields = Evaluator.__dataclass_fields__
    assert "dispatcher" not in fields
    assert "scheduler_core" not in fields


# ============================================================================
# Impl-9 meta — Run / __main__ / Q1 self-rescheduling / Q9 carry-over 영구 기록
# ============================================================================

def test_run_method_inventory():
    """Run 5 method bit-exact lock-in (Impl-9 Q2 · Q9 · Q10 정합)."""
    from puls_sched.run import Run
    publics = {
        n for n in dir(Run)
        if not n.startswith("_") and callable(getattr(Run, n))
    }
    expected = {"init", "step", "loop", "teardown", "main"}
    assert expected.issubset(publics)


def test_run_init_signature():
    """Run.init signature lock-in (Q3 dotted-path config + Q5 synthetic sentinel + Q9 named flag)."""
    from puls_sched.run import Run
    sig = inspect.signature(Run.init)
    params = list(sig.parameters.keys())
    assert params == ["config_module", "trace_path_or_synthetic", "output_dir", "seed"]


def test_run_main_signature():
    """Run.main signature lock-in (Q2 — `python -m puls_sched`)."""
    from puls_sched.run import Run
    sig = inspect.signature(Run.main)
    params = list(sig.parameters.keys())
    assert params == ["argv"]


def test_trace_synthesize_signature():
    """TraceReplayer.synthesize signature lock-in (Q5 — Impl-9 acceptance source)."""
    from puls_sched.trace import TraceReplayer
    sig = inspect.signature(TraceReplayer.synthesize)
    params = list(sig.parameters.keys())
    assert params == ["n", "seed"]


def test_admission_config_has_tick_interval_us():
    """AdmissionConfig.tick_interval_us field — Q1 self-rescheduling cadence placeholder."""
    from puls_sched.config import AdmissionConfig
    fields = AdmissionConfig.__dataclass_fields__
    assert "tick_interval_us" in fields
    assert fields["tick_interval_us"].type is float


def test_scheduler_core_has_self_rescheduling_flag():
    """SchedulerCore.enable_admission_tick_rescheduling — Impl-9 opt-in flag (R14)."""
    from puls_sched.main_loop import SchedulerCore
    fields = SchedulerCore.__dataclass_fields__
    assert "enable_admission_tick_rescheduling" in fields
    assert fields["enable_admission_tick_rescheduling"].type is bool


def test_scheduler_core_has_schedule_next_admission_tick():
    """SchedulerCore._schedule_next_admission_tick method (Q1 chain entry)."""
    from puls_sched.main_loop import SchedulerCore
    assert hasattr(SchedulerCore, "_schedule_next_admission_tick")
    sig = inspect.signature(SchedulerCore._schedule_next_admission_tick)
    params = list(sig.parameters.keys())
    assert params == ["self", "prev_event"]


def test_dag_has_reset_micro_batch_method():
    """DAG.reset_micro_batch method (Impl-9 ARCH §3.4 L × cycle re-dispatch entry)."""
    from puls_sched.dag import DAG
    assert hasattr(DAG, "reset_micro_batch")
    sig = inspect.signature(DAG.reset_micro_batch)
    params = list(sig.parameters.keys())
    assert params == ["self", "micro_batch_id"]


def test_window_has_evict_method():
    """InFlightWindow.evict method (Q9 carry-over 해소 — explicit mb eviction)."""
    from puls_sched.window import InFlightWindow
    assert hasattr(InFlightWindow, "evict")
    sig = inspect.signature(InFlightWindow.evict)
    params = list(sig.parameters.keys())
    assert params == ["self", "micro_batch_id"]


# ============================================================================
# Impl-9 R12 — ARCH §6.3 priority literal 영구 lock-in
# ============================================================================

_DISPATCHER_PRIORITY_LITERAL = ("O_PROJ", "PREFILL_ATTN", "QKV")


def test_dispatcher_gpu_priority_order_literal_lock_in():
    """ARCH §6.3 GPU priority literal — O_PROJ > PREFILL_ATTN > QKV. R12 영구 lock-in.

    미래 priority 변경 시 *두 영역 (production code + Cluster E test) 동시 갱신 강제*.
    """
    from puls_sched.dispatcher import GPU_PRIORITY_ORDER
    actual = tuple(t.name for t in GPU_PRIORITY_ORDER)
    assert actual == _DISPATCHER_PRIORITY_LITERAL
