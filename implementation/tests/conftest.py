import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event_queue import EventQueue
from puls_sched.main_loop import SchedulerCore
from puls_sched.window import InFlightWindow


@pytest.fixture
def dummy_config():
    return default_dummy_config()


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
def dispatcher(dummy_config, clock, event_queue, dag):
    return Dispatcher(config=dummy_config, clock=clock, queue=event_queue, dag=dag)


@pytest.fixture
def scheduler_core(dummy_config, clock, event_queue, dag, window, dispatcher):
    return SchedulerCore(
        config=dummy_config,
        clock=clock,
        queue=event_queue,
        dag=dag,
        window=window,
        dispatcher=dispatcher,
    )
