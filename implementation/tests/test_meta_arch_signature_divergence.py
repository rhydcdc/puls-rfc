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
