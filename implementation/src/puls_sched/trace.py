"""Long-ctx production trace ingest + replay + stats (Impl-6).

PLAN.md §4 Impl-6 — Vidur 변환 long-ctx trace 의 schema 정합 ingest + 분포 sanity.
Q1 — implementation/data/ 의 CSV 그대로 사용 (외부 path 의존 0).
Q2 — CSV 1 schema 만 지원 (longbench_csv). YAGNI.
Q4 — load → replay 의 자기 일치 (KS test p > 0.05 자명) + stats() 의 regression detection.
Q8 — 1M-class · mid-ctx 는 NotImplementedError("Phase 3") stub.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from puls_sched.request import Request


@dataclass(frozen=True)
class TraceEntry:
    """Long-ctx production trace 의 한 row.

    Q2 — CSV schema 그대로: arrived_at · num_prefill_tokens · num_decode_tokens.
    Q3 — 출처 = LongBench, arrival = 사용자 추가 Poisson(λ) synthetic.
    """

    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int


@dataclass(frozen=True)
class TraceStats:
    """Trace 분포 통계 — KS test 입력 + R1 regression detection 의 lock-in 대상."""

    n_entries: int
    kv_length_min: int
    kv_length_max: int
    kv_length_mean: float
    arrival_interval_mean: float
    arrival_interval_std: float


_SUPPORTED_SCHEMAS = ("longbench_csv",)
_PHASE_3_SCHEMAS = ("vidur_1m_class", "mooncake_chat")


@dataclass
class TraceReplayer:
    """Long-ctx production trace 의 ingest + replay + stats.

    Stateless 산식 — entries 가 frozen tuple, replay 는 generator (다회 호출 시 fresh).
    RNG 의존 0 → seed independence 자연 보존 (PLAN §0 C5 + ARCH §3.5.2).
    """

    entries: tuple[TraceEntry, ...] = field(default_factory=tuple)

    @staticmethod
    def load(path: Path | str, schema: str = "longbench_csv") -> "TraceReplayer":
        """CSV ingest 또는 Phase 3 stub raise.

        Raises:
            NotImplementedError: Phase 3 schema (Q8).
            ValueError: unsupported schema · empty CSV · missing field · wrong type · negative value.
            FileNotFoundError: path 미존재.
        """
        if schema in _PHASE_3_SCHEMAS:
            raise NotImplementedError(
                f"schema '{schema}' is Phase 3 — see Impl-10 (PLAN.md §4 Impl-6 'Phase 3 영역')"
            )
        if schema not in _SUPPORTED_SCHEMAS:
            raise ValueError(
                f"unsupported schema '{schema}', supported: {_SUPPORTED_SCHEMAS}"
            )
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"trace file not found: {p}")
        entries: list[TraceEntry] = []
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"arrived_at", "num_prefill_tokens", "num_decode_tokens"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"CSV header missing required fields {required}, got {reader.fieldnames}"
                )
            for i, row in enumerate(reader):
                try:
                    arrived = float(row["arrived_at"])
                    n_prefill = int(row["num_prefill_tokens"])
                    n_decode = int(row["num_decode_tokens"])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"CSV row {i}: type error — {e}")
                if arrived < 0 or n_prefill < 0 or n_decode < 0:
                    raise ValueError(
                        f"CSV row {i}: negative value "
                        f"(arrived={arrived}, prefill={n_prefill}, decode={n_decode})"
                    )
                entries.append(TraceEntry(arrived, n_prefill, n_decode))
        if not entries:
            raise ValueError(f"trace file is empty: {p}")
        return TraceReplayer(entries=tuple(entries))

    def replay(self, rate_multiplier: float = 1.0) -> Iterator[Request]:
        """Trace entries → Request generator (fresh per call).

        Q6 (c) — max_tokens = num_decode_tokens (trace-driven count path).
        ARCH §3.3 — kv_length = num_prefill + num_decode (full reservation).

        Raises:
            ValueError: rate_multiplier ≤ 0.
        """
        if rate_multiplier <= 0:
            raise ValueError(f"rate_multiplier must be positive, got {rate_multiplier}")
        for req_id, entry in enumerate(self.entries):
            yield Request(
                id=req_id,
                prompt_tokens=[0] * entry.num_prefill_tokens,
                kv_length=entry.num_prefill_tokens + entry.num_decode_tokens,
                arrival_time=entry.arrived_at / rate_multiplier,
                max_tokens=entry.num_decode_tokens,
            )

    def stats(self) -> TraceStats:
        """Trace 분포 통계 산출. Raises RuntimeError 시 entries 빈 상태."""
        if not self.entries:
            raise RuntimeError("stats() called on empty TraceReplayer")
        kv_lens = [e.num_prefill_tokens + e.num_decode_tokens for e in self.entries]
        if len(self.entries) > 1:
            intervals = [
                self.entries[i + 1].arrived_at - self.entries[i].arrived_at
                for i in range(len(self.entries) - 1)
            ]
            interval_mean = sum(intervals) / len(intervals)
            interval_var = sum((x - interval_mean) ** 2 for x in intervals) / len(intervals)
            interval_std = interval_var ** 0.5
        else:
            interval_mean = 0.0
            interval_std = 0.0
        return TraceStats(
            n_entries=len(self.entries),
            kv_length_min=min(kv_lens),
            kv_length_max=max(kv_lens),
            kv_length_mean=sum(kv_lens) / len(kv_lens),
            arrival_interval_mean=interval_mean,
            arrival_interval_std=interval_std,
        )
