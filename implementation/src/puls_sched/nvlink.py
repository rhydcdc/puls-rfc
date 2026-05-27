from dataclasses import dataclass

from puls_sched.config import Config


@dataclass(frozen=True)
class NVLinkTransfer:
    """A ↔ B inter-instance NVLink transfer 시간 산출기 (stateless, pure function).

    ARCH §3.4 정합:
    - A → B: O projection output [B × hidden]
    - B → A: FFN output [B × hidden] (next layer input)
    - "Asynchronous transfer allows hiding within A/B computation time" — data path,
      dispatched resource 아님. Event push · 자원 lock 안 함 (Q4).

    bytes_per_element 는 hidden state precision (default FP16 = 2 bytes).
    """

    config: Config
    bytes_per_element: int = 2

    def time(self, tensor_shape: tuple[int, ...]) -> float:
        """Tensor shape 위 NVLink transfer time 산출.

        Args:
            tensor_shape: e.g. ``(B, hidden)`` decode, ``(B * chunk, hidden)`` prefill.

        Returns:
            transfer time (ns) = bytes(shape) × config.time.nvlink_time_per_byte_ns.
        """
        if not tensor_shape:
            raise ValueError("tensor_shape must not be empty")
        for dim in tensor_shape:
            if dim < 0:
                raise ValueError(f"tensor_shape dim must be non-negative, got {dim}")
        n_elements = 1
        for dim in tensor_shape:
            n_elements *= dim
        n_bytes = n_elements * self.bytes_per_element
        return n_bytes * self.config.time.nvlink_time_per_byte_ns
