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
    """op_time signature = {self, k_channels, kv_rows_total}. N_decode 부재 (ARCH §3.1)."""
    params = set(inspect.signature(PIMExecutor.op_time).parameters.keys())
    assert params == {"self", "k_channels", "kv_rows_total"}, (
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
