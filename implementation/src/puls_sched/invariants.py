from puls_sched.dag import DAG
from puls_sched.node import NodeState, NodeType


def check_I1(dag: DAG, micro_batch_id: int) -> None:
    """I1 (correctness): prefill-attn(X) requires QKV(X) done."""
    qkv = dag.get_node(micro_batch_id, NodeType.QKV)
    if qkv.state is not NodeState.DONE:
        raise ValueError(
            f"I1 violation: prefill-attn({micro_batch_id}) dispatched "
            f"before QKV({micro_batch_id}) done (QKV state={qkv.state.name})"
        )


def check_I2(dag: DAG, micro_batch_id: int) -> None:
    """I2 (correctness): decode-attn(X) requires QKV(X) done."""
    qkv = dag.get_node(micro_batch_id, NodeType.QKV)
    if qkv.state is not NodeState.DONE:
        raise ValueError(
            f"I2 violation: decode-attn({micro_batch_id}) dispatched "
            f"before QKV({micro_batch_id}) done (QKV state={qkv.state.name})"
        )


def check_I3(dag: DAG, micro_batch_id: int) -> None:
    """I3 (efficiency): O-proj(X) requires prefill-attn(X) AND decode-attn(X) done."""
    pre = dag.get_node(micro_batch_id, NodeType.PREFILL_ATTN)
    dec = dag.get_node(micro_batch_id, NodeType.DECODE_ATTN)
    if pre.state is not NodeState.DONE or dec.state is not NodeState.DONE:
        raise ValueError(
            f"I3 violation: O-proj({micro_batch_id}) dispatched before "
            f"prefill-attn(state={pre.state.name}) AND decode-attn(state={dec.state.name}) done"
        )


def check_I4(gpu_busy: bool) -> None:
    """I4 (resource): only one GPU GEMM/attention op at time t."""
    if gpu_busy:
        raise ValueError("I4 violation: GPU already running an op")


def check_I5(pim_busy: bool) -> None:
    """I5 (resource): only one PIM decode-attn op at time t."""
    if pim_busy:
        raise ValueError("I5 violation: PIM already running an op")
