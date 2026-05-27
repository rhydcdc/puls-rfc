from dataclasses import dataclass


@dataclass
class Instance:
    """Instance A 또는 B 의 자원 추적기. TP=8 lock-step (ARCH §3.4) — GPU 1 자원 단위.

    Instance A — has_pim=True (SP-PIM 보유, ARCH §3.4)
    Instance B — has_pim=False (post-attention FFN only, no PIM)
    """

    name: str
    has_pim: bool
    gpu_busy: bool = False
    pim_busy: bool = False

    def acquire_gpu(self) -> None:
        if self.gpu_busy:
            raise RuntimeError(f"Instance {self.name}: GPU already busy")
        self.gpu_busy = True

    def release_gpu(self) -> None:
        if not self.gpu_busy:
            raise RuntimeError(f"Instance {self.name}: GPU not busy (double release)")
        self.gpu_busy = False

    def acquire_pim(self) -> None:
        if not self.has_pim:
            raise RuntimeError(f"Instance {self.name}: no PIM (ARCH §3.4 — Instance B no PIM)")
        if self.pim_busy:
            raise RuntimeError(f"Instance {self.name}: PIM already busy")
        self.pim_busy = True

    def release_pim(self) -> None:
        if not self.has_pim:
            raise RuntimeError(f"Instance {self.name}: no PIM (ARCH §3.4)")
        if not self.pim_busy:
            raise RuntimeError(f"Instance {self.name}: PIM not busy (double release)")
        self.pim_busy = False
