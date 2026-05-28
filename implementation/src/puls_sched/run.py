"""End-to-end driver (Impl-9). §0 C1~C5 completeness checkpoint substrate.

PLAN.md §4 Impl-9 + §0 Completeness Definition 정합. D1 (동작하는 scheduler) 완성 시점.

Run 의 책임:
1. init — Module instantiate + evaluator wiring + trace eager pre-load + first ADMISSION_TICK priming
2. step — SchedulerCore.step() delegate (Q10 thin wrapper)
3. loop — termination 3 조건 (queue empty AND window empty AND in_flight empty) 까지 step 반복
4. teardown — evaluator.report() → JSON + Markdown 산출 (Q8)
5. main — argv 파싱 → init → loop → teardown → exit code
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import Config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.evaluator import Evaluator
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.forward_pass import LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.nvlink import NVLinkTransfer
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request_queue import RequestQueue
from puls_sched.trace import TraceReplayer
from puls_sched.window import InFlightWindow


_TRACE_SYNTHETIC_PREFIX = "synthetic:"


def _to_jsonable(obj):
    """dataclass · enum · tuple → JSON-serializable. Recursive."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


@dataclass
class Run:
    """End-to-end driver. ARCH literal + PLAN §0 C1~C5 정합."""

    config: Config
    scheduler: SchedulerCore
    evaluator: Evaluator
    output_dir: Path
    _step_count: int = 0
    # Safety guard — Stage 2 위 per-mb spec-derived 산출 위 chunk 작아지면 step 수 자연 증가.
    # synthetic:30 위 평균 prompt ~500 + max_decode ~30 + chunk_per_req ~100 → 5 mb prefill +
    # 30 mb decode × L=80 × ~6 event/cycle ≈ 200K event/req × 30 req ≈ 6M. 100M wide margin.
    _safety_step_limit: int = 100_000_000

    @staticmethod
    def init(
        config_module: str,
        trace_path_or_synthetic: str,
        output_dir: Path | str,
        seed: int | None = None,
    ) -> "Run":
        """모든 module instantiate + wiring + trace eager pre-load (Q4)."""
        # Config — dotted-path lookup (Q3)
        if ":" not in config_module:
            raise ValueError(
                f"config_module must be 'module.path:factory_fn', got {config_module!r}"
            )
        mod_path, factory_name = config_module.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        factory = getattr(mod, factory_name)
        config: Config = factory()
        if seed is not None:
            config = dataclasses.replace(config, seed=seed)

        # Trace (Q4 eager + Q5 synthesize sentinel)
        if trace_path_or_synthetic.startswith(_TRACE_SYNTHETIC_PREFIX):
            n_str = trace_path_or_synthetic[len(_TRACE_SYNTHETIC_PREFIX):]
            replayer = TraceReplayer.synthesize(n=int(n_str), seed=config.seed)
        else:
            replayer = TraceReplayer.load(Path(trace_path_or_synthetic))

        # Module instantiate (Impl-1~6+8 누적 정합)
        clock = Clock()
        queue = EventQueue(clock)
        dag = DAG()
        window = InFlightWindow(dag, config=config)
        idle_telemetry = IdleTelemetry()
        idle_telemetry.reset(t_start=0.0)
        pim_executor = PIMExecutor(config=config)
        dispatcher = Dispatcher(
            config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim_executor,
            idle_telemetry=idle_telemetry,    # Impl-10-pre-1 O8.2 wiring
        )
        request_queue = RequestQueue(capacity=config.admission.request_queue_capacity)
        kv_accountant = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
        admission = Admission(
            admission_cfg=config.admission,
            request_queue=request_queue,
            kv_accountant=kv_accountant,
            idle_telemetry=idle_telemetry,
        )
        layer_state = LayerState(num_layers=config.model.num_layers)
        completion = Completion(clock=clock, kv_accountant=kv_accountant)
        # Impl-10-pre-1 (A) — InstancePipeline plumbing 위 production hot path inter-AB chain
        instance_a = Instance(name="A", has_pim=True)
        instance_b = Instance(name="B", has_pim=False)
        nvlink = NVLinkTransfer(config=config)
        instance_pipeline = InstancePipeline(
            config=config, instance_a=instance_a, instance_b=instance_b, nvlink=nvlink,
            clock=clock, idle_telemetry=idle_telemetry,
        )
        scheduler = SchedulerCore(
            config=config, clock=clock, queue=queue, dag=dag, window=window,
            dispatcher=dispatcher, request_queue=request_queue,
            kv_accountant=kv_accountant, admission=admission,
            layer_state=layer_state, completion=completion,
            instance_pipeline=instance_pipeline,    # (A) production inter-AB wiring
        )
        # Q1 — self-rescheduling enable (R14 opt-in flag)
        scheduler.enable_admission_tick_rescheduling = True

        # Evaluator wiring (D3 standalone — Impl-8 hook API 정합)
        evaluator = Evaluator(config=config, clock=clock, idle_telemetry=idle_telemetry)
        dispatcher.on_dispatch(evaluator.record_dispatch)
        scheduler.on_admission_tick(evaluator.record_admission_tick)

        # Q4 eager pre-load — Trace REQUEST_ARRIVAL 전부 push
        for req in replayer.replay(rate_multiplier=1.0):
            queue.push(Event(
                timestamp=req.arrival_time,
                type=EventType.REQUEST_ARRIVAL,
                payload={"request": req},
            ))

        # Q1 first ADMISSION_TICK priming. Impl-10-pre-1 (B)~(B''') — composer 사용
        # (init 시점 in_flight empty + idle_telemetry 0 → 자연 trivial payload, 단 t_pim_fn 은 진정 closure).
        queue.push(Event(
            timestamp=0.0,
            type=EventType.ADMISSION_TICK,
            payload=scheduler._compose_admission_payload(),
        ))

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        return Run(config=config, scheduler=scheduler, evaluator=evaluator, output_dir=out)

    def step(self) -> bool:
        """Q10 — SchedulerCore.step() delegate. 1 event pop = 1 step."""
        progressed = self.scheduler.step()
        if progressed:
            self._step_count += 1
            if self._step_count > self._safety_step_limit:
                raise RuntimeError(
                    f"Run.loop exceeded safety_step_limit {self._safety_step_limit} "
                    f"(infinite loop suspected — see Impl-9 R3)"
                )
        return progressed

    def loop(self) -> int:
        """Termination 3 조건 (queue empty AND window empty AND in_flight empty) 까지.

        Impl-10-pre-1 — `PULS_PROGRESS_INTERVAL` env (default 0 = off) 위 진행 logging.
        Set to N > 0 → 매 N step 마다 stderr 위 step/clock/in_flight/queue 출력.
        """
        progress_interval = int(os.environ.get("PULS_PROGRESS_INTERVAL", "0"))
        while True:
            if (
                len(self.scheduler.queue) == 0
                and len(self.scheduler.window.current_ids()) == 0
                and len(self.scheduler.in_flight_requests) == 0
            ):
                break
            if not self.step():
                break  # defensive
            if progress_interval > 0 and self._step_count % progress_interval == 0:
                print(
                    f"[Run.loop] step={self._step_count} "
                    f"clock={self.scheduler.clock.now:.1f} "
                    f"in_flight={len(self.scheduler.in_flight_requests)} "
                    f"queue={len(self.scheduler.queue)} "
                    f"window={len(self.scheduler.window.current_ids())}",
                    file=sys.stderr, flush=True,
                )
        return self._step_count

    def teardown(self) -> dict:
        """evaluator.report() → JSON + Markdown (Q8). Returns report dict.

        Atomicity (R8) — JSON 직렬화 + write 가 같은 try block.
        Cycle values 는 PLAN §0.5 dummy placeholder (Impl-10 calibrated 영역).
        """
        t_proj = max(self.config.time.gpu_op_time_us.values())
        t_pim = self.config.time.pim_tile_time_ns[self.config.model.kv_precision] / 1000.0
        # Ensure positive for pipeline_efficiency contract
        a_cycle = max(t_proj + t_pim, 1e-9)
        b_cycle = max(t_proj, 1e-9)
        report = self.evaluator.report(
            a_cycle=a_cycle, b_cycle=b_cycle, t_pim=t_pim, t_proj=t_proj,
        )
        report_for_json = {k: v for k, v in report.items() if k != "markdown"}
        json_text = json.dumps(_to_jsonable(report_for_json), sort_keys=True, indent=2)
        md_text = report["markdown"]
        (self.output_dir / "report.json").write_text(json_text, encoding="utf-8")
        (self.output_dir / "report.md").write_text(md_text, encoding="utf-8")
        return report

    @staticmethod
    def main(argv: list[str] | None = None) -> int:
        """Q2 + Q9 entry — argparse named flag."""
        parser = argparse.ArgumentParser(prog="puls-sched", description="PULS scheduler driver")
        parser.add_argument("--config-module", required=True,
                            help="dotted path 'module:factory_fn'")
        parser.add_argument("--trace", required=True,
                            help="CSV path or 'synthetic:N'")
        parser.add_argument("--output", required=True, help="output directory")
        parser.add_argument("--seed", type=int, default=None, help="override config.seed")
        ns = parser.parse_args(argv)
        try:
            run = Run.init(
                config_module=ns.config_module,
                trace_path_or_synthetic=ns.trace,
                output_dir=ns.output,
                seed=ns.seed,
            )
            run.loop()
            run.teardown()
        except (ValueError, FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
            print(f"Run failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        return 0
