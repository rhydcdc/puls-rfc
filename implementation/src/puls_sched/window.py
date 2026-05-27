import collections

from puls_sched.config import Config
from puls_sched.dag import DAG


class InFlightWindow:
    """3-μ-batch sliding window. ARCH §6.7 정합.

    Impl-8 — F2 ablation 위 config-driven capacity. F2 비활성화 시 capacity=1 강제로
    double-buffering 무효화 (μ-batch 직렬 강제, ARCH §5.7 F2 정의).
    """

    DEFAULT_CAPACITY = 3                              # ARCH §6.7 — 3-μ-batch 기본 (Impl-1 보존)

    def __init__(self, dag: DAG, config: Config | None = None):
        self._dag = dag
        # Impl-8 — config.ablation.f2_window_capacity_override 우선 lookup
        if config is not None and config.ablation.f2_window_capacity_override is not None:
            self.capacity = config.ablation.f2_window_capacity_override
        else:
            self.capacity = self.DEFAULT_CAPACITY
        if self.capacity < 1:
            raise ValueError(f"window capacity must be >= 1, got {self.capacity}")
        self._micro_batch_ids: collections.deque[int] = collections.deque(
            maxlen=self.capacity
        )

    def admit(self, micro_batch_id: int) -> int | None:
        evicted = None
        if len(self._micro_batch_ids) == self.capacity:
            evicted = self._micro_batch_ids[0]
            self._dag.remove_micro_batch(evicted)
        self._micro_batch_ids.append(micro_batch_id)
        self._dag.add_micro_batch(micro_batch_id)
        return evicted

    def evict(self, micro_batch_id: int) -> None:
        """Impl-9 — specific mb 의 explicit eviction (모든 req finalize 후).

        Q9 carry-over (MODULES.md Impl-6 'window eviction 은 Impl-9 영역') 해소.
        DAG remove 도 atomic 으로 수행. 미존재 mb_id 는 KeyError raise.
        """
        if micro_batch_id not in self._micro_batch_ids:
            raise KeyError(f"micro_batch {micro_batch_id} not in window (evict failed)")
        self._micro_batch_ids.remove(micro_batch_id)
        self._dag.remove_micro_batch(micro_batch_id)

    def current_ids(self) -> tuple[int, ...]:
        return tuple(self._micro_batch_ids)
