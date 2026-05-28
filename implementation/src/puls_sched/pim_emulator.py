"""PIM Executor Emulator.

Stateless 시간 계산기. ARCH §3.5.2 Computed Wait 정합 — dispatch agent 아님.
dispatcher 가 KERNEL_COMPLETION event 의 timestamp 산출 시 `pim_executor.op_time(...)`
호출 → `clock.now + op_time` push. PIMExecutor 자체는 clock · queue 모름.

ARCH 정합 근거:
- §3.1 "FSM cycle structure is invariant whether GEMV (B=1) or GEMM (B>1)" →
  batch dim (N_decode) 시간 영향 0 → signature 에 batch arg 부재
- §3.1 "FP16 MAC core / FP8 (E4M3) KV-cache storage" → regime = system-wide
  KV storage precision → `self.config.model.kv_precision` lookup
- §3.4 "KV rows are sharded across channels" → per-channel work = total / k
- §3.5.2 "No separate synchronization mechanism required — completion notification
  · interrupt · barrier all unnecessary" → dispatch method 부재 (pure calculator)
- §3.2 "256 channels per-GPU" → cross-GPU boundary = num_stacks × num_channels
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from puls_sched.config import Config


@dataclass(frozen=True)
class PIMExecutor:
    config: Config

    def tile_time(self) -> float:
        """tile-level time 산출. ARCH §3.1 regime 분기 + config.model.kv_precision lookup.

        - FP8 KV: compute-bound (tile load < compute, FSM cycle dominant)
        - FP16 KV: load-bound (tile load > compute, ≈ 2× FP8 — ARCH §6.6)

        값 자체는 placeholder (PLAN §0.5). Ratio property 만 dummy 가 보존.
        """
        regime = self.config.model.kv_precision
        return self.config.time.pim_tile_time_ns[regime]

    @property
    def k_aggregate(self) -> int:
        """Total SP-PIM channels across Instance A (ARCH §3.2 + §3.4 literal).

        = num_gpus_instance_a × num_stacks_per_gpu × num_channels_per_stack.
        Sequence-parallel PIM 위 channel 영역 micromanagement 불필요 — 임의 시점 한 mb 의
        decode-attn 이 모든 채널 점유 (Impl-10-pre-2 — k_total knob 폐기).
        """
        hw = self.config.hw
        return hw.num_gpus_instance_a * hw.num_stacks_per_gpu * hw.num_channels_per_stack

    def op_time(self, kv_rows_total: int, kv_rows_lockstep: int = 0) -> float:
        """SP-PIM aggregate op-level time. ARCH §3.4 Q-replicate + KV-row sharding.

        Args:
            kv_rows_total: batch 의 KV row 합 (exact, F5 활성화 path 위 산식 입력).
            kv_rows_lockstep: max(kv_length) × num_decode_reqs (F5 ablation 위 lock-step
                              penalty 산식 입력). F5 활성화 path 위에선 사용 안 함 (default 0).

        Returns:
            op time = per-channel tile count × tile_time (broadcast overhead 영원 cross-GPU 적용).

        산식 (F5 활성화 — 정상 PULS 동작):
            tiles_per_channel = ceil(kv_rows_total / (k_aggregate × rtl_fsm_tile_rows))

        산식 (F5 비활성화 — ARCH §5.7 F5 "max-KV straggler bubble within a batch"):
            tiles_per_channel = ceil(kv_rows_lockstep / (k_aggregate × rtl_fsm_tile_rows))

        Impl-10-pre-2 — k_channels 매개변수 폐기 (sequence-parallel PIM 위 무의미, k_total knob 제거).
        """
        if kv_rows_total <= 0 and kv_rows_lockstep <= 0:
            return 0.0  # decode rows 0 — pure-prefill batch (ARCH §5.1)
        k_channels = self.k_aggregate
        tile_rows = self.config.time.rtl_fsm_tile_rows
        if self.config.ablation.f5_disabled:
            if kv_rows_lockstep <= 0:
                raise ValueError(
                    f"F5 ablation requires kv_rows_lockstep > 0, got {kv_rows_lockstep} "
                    f"(MicroBatch.kv_rows_lockstep 가 admission 에서 산출되어야 함)"
                )
            tiles_per_channel = math.ceil(kv_rows_lockstep / (k_channels * tile_rows))
        else:
            tiles_per_channel = math.ceil(kv_rows_total / (k_channels * tile_rows))
        per_channel_time = tiles_per_channel * self.tile_time()
        # Cross-GPU broadcast overhead — k_aggregate 영원 per-GPU max 초과 (Instance A 8 GPUs)
        broadcast = self.config.time.pim_broadcast_latency_ns_cross_gpu
        return per_channel_time + broadcast

    @staticmethod
    def load_ramulator2_cycles(path: Path) -> Mapping[str, int]:
        """Ramulator2 HBM4 추정 cycle JSON ingest. Impl-4 단계는 fixture dummy.

        Schema: list of {"regime": "FP8"|"FP16", "tile_cycle": int, "fsm_freq_ghz": float}
        Returns: {regime: tile_cycle} dict.

        Fail-fast on FileNotFoundError / malformed schema (empty, missing field,
        wrong type, negative cycle, duplicate regime).

        실데이터 ingest 는 Impl-10 영역 (PLAN.md §0.5).
        """
        if not path.exists():
            raise FileNotFoundError(f"Ramulator2 cycles file not found: {path}")
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not data:
            raise ValueError(
                f"Ramulator2 cycles must be non-empty list, got "
                f"{type(data).__name__} (len={len(data) if hasattr(data, '__len__') else 'n/a'})"
            )
        out: dict[str, int] = {}
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Ramulator2 entry must be dict, got {type(entry).__name__}"
                )
            for field in ("regime", "tile_cycle", "fsm_freq_ghz"):
                if field not in entry:
                    raise ValueError(
                        f"Ramulator2 entry missing field '{field}': {entry}"
                    )
            regime = entry["regime"]
            if not isinstance(regime, str):
                raise ValueError(
                    f"Ramulator2 entry 'regime' must be str, got "
                    f"{type(regime).__name__}: {entry}"
                )
            tile_cycle = entry["tile_cycle"]
            if not isinstance(tile_cycle, int) or isinstance(tile_cycle, bool) or tile_cycle < 0:
                raise ValueError(
                    f"Ramulator2 entry 'tile_cycle' must be non-negative int: {entry}"
                )
            if regime in out:
                raise ValueError(f"Ramulator2 duplicate regime entry: {regime}")
            out[regime] = tile_cycle
        return out
