import dataclasses

import pytest

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event_queue import EventQueue
from puls_sched.forward_pass import ForwardPass, LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.nvlink import NVLinkTransfer
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


@pytest.fixture
def dummy_config():
    return default_dummy_config()


@pytest.fixture
def admission_config(dummy_config):
    return dummy_config.admission


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def event_queue(clock):
    return EventQueue(clock)


@pytest.fixture
def dag():
    return DAG()


@pytest.fixture
def window(dag):
    return InFlightWindow(dag)


@pytest.fixture
def pim_executor(dummy_config):
    return PIMExecutor(config=dummy_config)


@pytest.fixture
def pim_executor_fp16(dummy_config):
    """Cross-product fixture — kv_precision="FP16" 변종 (Q1 system-wide regime)."""
    model_fp16 = dataclasses.replace(dummy_config.model, kv_precision="FP16")
    cfg_fp16 = dataclasses.replace(dummy_config, model=model_fp16)
    return PIMExecutor(config=cfg_fp16)


@pytest.fixture
def dispatcher(dummy_config, clock, event_queue, dag, pim_executor):
    return Dispatcher(
        config=dummy_config,
        clock=clock,
        queue=event_queue,
        dag=dag,
        pim_executor=pim_executor,
    )


@pytest.fixture
def request_queue(admission_config):
    return RequestQueue(capacity=admission_config.request_queue_capacity)


@pytest.fixture
def kv_accountant(admission_config):
    return KVAccountant(capacity=admission_config.kv_capacity_aggregate)


@pytest.fixture
def idle_telemetry():
    return IdleTelemetry()


@pytest.fixture
def admission(admission_config, request_queue, kv_accountant, idle_telemetry):
    return Admission(
        admission_cfg=admission_config,
        request_queue=request_queue,
        kv_accountant=kv_accountant,
        idle_telemetry=idle_telemetry,
    )


@pytest.fixture
def completion(clock, kv_accountant):
    return Completion(clock=clock, kv_accountant=kv_accountant)


@pytest.fixture
def scheduler_core(dummy_config, clock, event_queue, dag, window, dispatcher,
                   request_queue, kv_accountant, admission, layer_state, completion):
    return SchedulerCore(
        config=dummy_config,
        clock=clock,
        queue=event_queue,
        dag=dag,
        window=window,
        dispatcher=dispatcher,
        request_queue=request_queue,
        kv_accountant=kv_accountant,
        admission=admission,
        layer_state=layer_state,
        completion=completion,
    )


# =========================================================================
# Impl-5 fixtures
# =========================================================================

@pytest.fixture
def instance_a():
    return Instance(name="A", has_pim=True)


@pytest.fixture
def instance_b():
    return Instance(name="B", has_pim=False)


@pytest.fixture
def nvlink_transfer(dummy_config):
    return NVLinkTransfer(config=dummy_config)


@pytest.fixture
def instance_pipeline(dummy_config, instance_a, instance_b, nvlink_transfer):
    return InstancePipeline(
        config=dummy_config,
        instance_a=instance_a,
        instance_b=instance_b,
        nvlink=nvlink_transfer,
    )


@pytest.fixture
def layer_state(dummy_config):
    return LayerState(num_layers=dummy_config.model.num_layers)


@pytest.fixture
def forward_pass(dummy_config, instance_pipeline, layer_state):
    return ForwardPass(
        config=dummy_config,
        instance_pipeline=instance_pipeline,
        layer_state=layer_state,
    )


# =========================================================================
# Impl-8 fixtures
# =========================================================================

from puls_sched.config import AblationConfig
from puls_sched.evaluator import Evaluator


@pytest.fixture
def ablation_config_default():
    """Default ablation config — 모든 flag off (정상 PULS 동작 backward-compat)."""
    return AblationConfig()


@pytest.fixture
def ablation_config_all_disabled():
    """모든 F1·F2·F3·F5 disabled (composite ablation, Impl-10 sweep 영역과 형태만 동일)."""
    return AblationConfig(
        f1_disabled=True,
        f2_window_capacity_override=1,
        f3_disabled=True,
        f5_disabled=True,
    )


@pytest.fixture
def config_f1_disabled(dummy_config):
    ab = dataclasses.replace(dummy_config.ablation, f1_disabled=True)
    return dataclasses.replace(dummy_config, ablation=ab)


@pytest.fixture
def config_f2_capacity_1(dummy_config):
    ab = dataclasses.replace(dummy_config.ablation, f2_window_capacity_override=1)
    return dataclasses.replace(dummy_config, ablation=ab)


@pytest.fixture
def config_f5_disabled(dummy_config):
    ab = dataclasses.replace(dummy_config.ablation, f5_disabled=True)
    return dataclasses.replace(dummy_config, ablation=ab)


@pytest.fixture
def evaluator(dummy_config, clock, idle_telemetry):
    return Evaluator(config=dummy_config, clock=clock, idle_telemetry=idle_telemetry)
