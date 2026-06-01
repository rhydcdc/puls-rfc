"""동작점 도출 스윕 — OPERATING_POINT.md §5 의 두 표를 재현.

ctx·prefill 을 바꿔가며 op-time(PIMExecutor·compute_ffn/gpu_op_time_s, TP=8)으로
삼중-균형점을 찾는다. 결론: ① ctx 100K 가 유일 균형점, ② 균형 ctx 는 prefill 무관(100K
하드웨어 상수), prefill 은 X(레이턴시·배치 규모)만 정하는 스케일 knob.

실행:
    cd implementation && PYTHONIOENCODING=utf-8 python debug_phase2/sweep_operating_point.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puls_sched.config import (
    default_dummy_config,
    compute_ffn_op_time_s,
    compute_gpu_op_time_s,
)
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType
from puls_sched.pim_emulator import PIMExecutor

CFG = default_dummy_config()
PIM = PIMExecutor(CFG)
NGPU = CFG.hw.num_gpus_instance_a  # TP=8 — GPU/FFN 시간 ÷num_gpus, PIM 은 k_aggregate(2048ch)


def three_times(n_dec: int, ctx: int, prefill: int, n_prefill_reqs: int = 8):
    """(t_PIM, t_GPU_A, t_FFN) µs. 단위: PIM op_time=ns(×1e-3), GPU/FFN=s(×1e6)."""
    chunk_each = max(prefill // n_prefill_reqs, 1)
    prefill_chunk = {10_000 + j: list(range(chunk_each)) for j in range(n_prefill_reqs)}
    prefill_processed = {10_000 + j: ctx for j in range(n_prefill_reqs)}  # 청크 깊이 = ctx
    kv = n_dec * ctx
    mb = MicroBatch(
        id=0, kv_rows_total=kv, kv_rows_lockstep=0,
        prefill_chunk=prefill_chunk, decode_tokens={i: 0 for i in range(max(n_dec, 0))},
        prefill_chunk_budget=prefill, prefill_processed=prefill_processed,
    )
    t_pim = PIM.op_time(kv_rows_total=kv) * 1e-3 if kv > 0 else 0.0
    t_gpu_a = sum(
        compute_gpu_op_time_s(nt, mb, CFG.calibration, CFG.model, num_gpus=NGPU) * 1e6
        for nt in (NodeType.QKV, NodeType.O_PROJ, NodeType.PREFILL_ATTN)
    )
    t_ffn = compute_ffn_op_time_s(mb, CFG.calibration, CFG.model, num_gpus=NGPU) * 1e6
    return t_pim, t_gpu_a, t_ffn


def balance_n_dec(ctx: int, prefill: int) -> int:
    """FFN = GPU-A 되는 N_dec (이진 탐색)."""
    lo, hi = 0, 8000
    for _ in range(40):
        mid = (lo + hi) // 2
        _, t_gpu_a, t_ffn = three_times(mid, ctx, prefill)
        if t_ffn < t_gpu_a:
            lo = mid + 1
        else:
            hi = mid
    return lo


def spread(times) -> float:
    hi = max(times)
    return (hi - min(times)) / hi * 100 if hi > 0 else 100.0


def sweep_a_ctx(prefill: int = 512):
    """스윕 A — ctx 고정 스윕(prefill 512): 100K 가 유일 삼중-균형점."""
    print(f"\n=== 스윕 A — ctx 스윕 (prefill={prefill}) ===")
    print(f'{"ctx":>8} {"N_dec*":>7} {"X(FFN=GPUa)":>12} {"t_PIM":>8} {"Σkv":>9} {"spread%":>8}')
    for ctx in (40_000, 60_000, 80_000, 100_000, 120_000, 150_000, 200_000, 300_000):
        n = balance_n_dec(ctx, prefill)
        t_pim, t_gpu_a, t_ffn = three_times(n, ctx, prefill)
        X = (t_gpu_a + t_ffn) / 2
        print(f'{ctx:8d} {n:7d} {X:11.1f}µs {t_pim:8.1f} {n*ctx/1e6:8.1f}M {spread((t_pim, t_gpu_a, t_ffn)):8.1f}')


def sweep_b_prefill():
    """스윕 B — prefill × ctx: 모든 prefill 이 ctx 100K 에서 균형(=하드웨어 상수)."""
    print("\n=== 스윕 B — prefill × ctx (각 prefill 의 삼중균형 ctx 탐색) ===")
    print(f'{"prefill":>7} {"균형ctx":>7} {"X(µs)":>7} {"N_dec":>6} {"decodeKV":>9} {"prefKVwork":>11} {"spread%":>8}')
    for prefill in (256, 512, 1024, 2048):
        best = None
        for ctx in range(20_000, 400_001, 2_000):
            n = balance_n_dec(ctx, prefill)
            t_pim, t_gpu_a, t_ffn = three_times(n, ctx, prefill)
            sp = spread((t_pim, t_gpu_a, t_ffn))
            if best is None or sp < best[-1]:
                best = (ctx, (t_gpu_a + t_ffn) / 2, n, n * ctx, prefill * ctx, sp)
        ctx, X, n, kv, pw, sp = best
        print(f'{prefill:7d} {ctx//1000:6d}K {X:7.0f} {n:6d} {kv/1e6:8.1f}M {pw/1e6:10.1f}M {sp:8.2f}')


if __name__ == "__main__":
    print(f"HW: instance_a={CFG.hw.num_gpus_instance_a} GPU, PIM k_aggregate={PIM.k_aggregate}ch, "
          f"peak(TP8)={CFG.calibration.gpu_fp16_dense_peak_tflops*1e12*CFG.calibration.gpu_mfu_default*NGPU:.3e} FLOPS")
    sweep_a_ctx(prefill=512)
    sweep_b_prefill()
