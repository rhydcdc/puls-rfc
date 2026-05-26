from puls_sched.dag import DAG
from puls_sched.window import InFlightWindow


def test_admit_under_capacity():
    window = InFlightWindow(DAG())
    assert window.admit(0) is None
    assert window.admit(1) is None
    assert window.admit(2) is None
    assert window.current_ids() == (0, 1, 2)


def test_admit_evicts_oldest():
    window = InFlightWindow(DAG())
    for i in range(3):
        window.admit(i)
    evicted = window.admit(3)
    assert evicted == 0
    assert window.current_ids() == (1, 2, 3)


def test_evict_syncs_dag():
    dag = DAG()
    window = InFlightWindow(dag)
    for i in range(3):
        window.admit(i)
    window.admit(3)  # evicts 0
    assert 0 not in dag.nodes
    assert 0 not in dag.precedence
    assert set(dag.nodes.keys()) == {1, 2, 3}


def test_admit_syncs_dag():
    dag = DAG()
    window = InFlightWindow(dag)
    window.admit(0)
    assert 0 in dag.nodes
    assert len(dag.nodes[0]) == 4  # 4 NodeType


def test_invariant_window_dag_consistency():
    for seq in [(0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5)]:
        dag = DAG()
        window = InFlightWindow(dag)
        for mid in seq:
            window.admit(mid)
        assert set(dag.nodes.keys()) == set(window.current_ids())
        assert set(dag.precedence.keys()) == set(window.current_ids())
