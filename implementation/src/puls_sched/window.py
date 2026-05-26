import collections

from puls_sched.dag import DAG


class InFlightWindow:
    CAPACITY = 3

    def __init__(self, dag: DAG):
        self._dag = dag
        self._micro_batch_ids: collections.deque[int] = collections.deque(
            maxlen=self.CAPACITY
        )

    def admit(self, micro_batch_id: int) -> int | None:
        evicted = None
        if len(self._micro_batch_ids) == self.CAPACITY:
            evicted = self._micro_batch_ids[0]
            self._dag.remove_micro_batch(evicted)
        self._micro_batch_ids.append(micro_batch_id)
        self._dag.add_micro_batch(micro_batch_id)
        return evicted

    def current_ids(self) -> tuple[int, ...]:
        return tuple(self._micro_batch_ids)
