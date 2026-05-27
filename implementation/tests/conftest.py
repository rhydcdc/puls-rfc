import dataclasses

import pytest

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event_queue import EventQueue
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
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
def scheduler_core(dummy_config, clock, event_queue, dag, window, dispatcher,
                   request_queue, kv_accountant, admission):
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
    )
