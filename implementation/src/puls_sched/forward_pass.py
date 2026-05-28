from dataclasses import dataclass

from puls_sched.config import Config
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.micro_batch import MicroBatch


@dataclass
class LayerState:
    """MicroBatch 의 forward pass 진행 상태 tracker.

    Q3 — 단순 index tracker. Dispatcher 는 layer-agnostic.
    """

    num_layers: int

    def advance(self, mb: MicroBatch) -> bool:
        """MicroBatch.current_layer_index 를 1 증가. L 도달 시 True (token decode signal trigger).

        Raises:
            ValueError: current_layer_index 가 0 미만 또는 num_layers 이상 (단조 위반).
        """
        if mb.current_layer_index < 0:
            raise ValueError(
                f"current_layer_index must be non-negative, got {mb.current_layer_index}"
            )
        if mb.current_layer_index >= self.num_layers:
            raise ValueError(
                f"current_layer_index ({mb.current_layer_index}) already reached "
                f"num_layers ({self.num_layers}) — forward pass already done"
            )
        mb.current_layer_index += 1
        return mb.current_layer_index >= self.num_layers


@dataclass
class ForwardPass:
    """L-layer iteration loop owner. ARCH §3.4 "Pass through L layers = L × cycle" 정합.

    Q3 — forward_pass = L-loop owner. InstancePipeline = 단일 layer cycle owner.

    Impl-5 의 run() 은 *iteration meta-count + token decode signal trigger* 만 검증.
    실 instance_pipeline.dispatch 통합은 Impl-9 driver 영역 (§7 O5.7).
    """

    config: Config
    instance_pipeline: InstancePipeline
    layer_state: LayerState

    def run(self, mb: MicroBatch) -> int:
        """MicroBatch 위 L-layer iteration 실행.

        Impl-10-pre-1 O5.1 — 매 layer 마다 `instance_pipeline.dispatch(mb)` 호출 (Q6 (a) 결정).
        ARCH §3.4 *forward pass = L × cycle* literal 정합 — 각 layer 가 A → handoff → B → handoff → A_next chain.

        Returns:
            count of layer advance (== num_layers if completed normally).
        """
        if mb.current_layer_index != 0:
            raise ValueError(
                f"forward pass entry: current_layer_index must be 0, got {mb.current_layer_index}"
            )
        count = 0
        while True:
            # O5.1 — 매 layer 의 A→B chain wiring (substrate 영역, ARCH §3.4 정합)
            self.instance_pipeline.dispatch(mb)
            done = self.layer_state.advance(mb)
            count += 1
            if done:
                return count
            if count > self.layer_state.num_layers:
                raise RuntimeError(
                    f"forward pass exceeded num_layers ({self.layer_state.num_layers})"
                )
