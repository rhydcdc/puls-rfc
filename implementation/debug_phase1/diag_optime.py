"""op_time 직접 측정 — ctx 70K 에서 PIM(decode-attn) vs GPU(prefill-attn·projection).

추측 중단. 시뮬레이터의 실제 per-op 시간을 직접 산출해 "GPU 가 왜 지배(PIM 유휴)"
하는지 규명. 시뮬 안 돌리고 compute_gpu_op_time_s + PIMExecutor.op_time 직접 호출.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "debug_phase1")
from puls_sched.config import default_dummy_config, compute_gpu_op_time_s
from puls_sched.node import NodeType
from puls_sched.pim_emulator import PIMExecutor

cfg = default_dummy_config()
pim = PIMExecutor(config=cfg)

CTX = 70_000
N_DEC = 27          # 순수 decode 구간: 27 요청 decode 중
N_PRE = 3           # 10% 아직 prefill (warmup 0.9 지점)
CHUNK_PER_PRE = 170  # balance 가 줄 법한 작은 청크 (가정)


def mk_mb(decode_ids, prefill_ids, chunk_per, ctx):
    decode_tokens = {i: 0 for i in decode_ids}
    prefill_chunk = {i: list(range(chunk_per)) for i in prefill_ids}
    prefill_processed = {i: ctx for i in prefill_ids}   # 이미 ctx 만큼 처리(고ctx prefill 후반)
    return SimpleNamespace(
        decode_tokens=decode_tokens,
        prefill_chunk=prefill_chunk,
        prefill_processed=prefill_processed,
    )


def gpu_us(node, mb):
    return compute_gpu_op_time_s(node, mb, cfg.calibration, cfg.model, cfg.time) * 1e6


print(f"=== op_time @ ctx={CTX} ===\n")

# --- (1) 순수 decode (N_DEC 요청, prefill 0) ---
mb_dec = mk_mb(range(N_DEC), [], 0, CTX)
t_qkv = gpu_us(NodeType.QKV, mb_dec)
t_oproj = gpu_us(NodeType.O_PROJ, mb_dec)
t_pim = pim.op_time(kv_rows_total=N_DEC * CTX) * 1e6
print(f"[순수 decode, {N_DEC} req]")
print(f"  GPU QKV     = {t_qkv:.3f} us")
print(f"  GPU O_PROJ  = {t_oproj:.3f} us")
print(f"  GPU proj 합 = {t_qkv + t_oproj:.3f} us")
print(f"  PIM decode-attn = {t_pim:.3f} us")
print(f"  → PIM/GPUproj = {t_pim / (t_qkv + t_oproj):.3f}  (>1 이면 PIM-bound)")
print()

# --- (2) prefill-attn 1 요청 (ctx 70K, chunk) ---
mb_pre1 = mk_mb([], [0], CHUNK_PER_PRE, CTX)
t_pattn1 = gpu_us(NodeType.PREFILL_ATTN, mb_pre1)
print(f"[prefill-attn 1 req, chunk={CHUNK_PER_PRE}, ctx={CTX}]")
print(f"  GPU PREFILL_ATTN = {t_pattn1:.3f} us")
print(f"  → 이 1 req prefill-attn 이 위 PIM decode-attn({t_pim:.1f}us)의 {t_pattn1/t_pim:.1f}배")
print()

# --- (3) 혼합 (warmup 0.9 지점: 27 decode + 3 prefill) ---
mb_mix = mk_mb(range(N_DEC), [100, 101, 102], CHUNK_PER_PRE, CTX)
t_qkv_m = gpu_us(NodeType.QKV, mb_mix)
t_oproj_m = gpu_us(NodeType.O_PROJ, mb_mix)
t_pattn_m = gpu_us(NodeType.PREFILL_ATTN, mb_mix)
t_gpu_total = t_qkv_m + t_oproj_m + t_pattn_m
print(f"[혼합 {N_DEC} decode + {N_PRE} prefill (warmup 0.9 지점)]")
print(f"  GPU QKV={t_qkv_m:.2f} O_PROJ={t_oproj_m:.2f} PREFILL_ATTN={t_pattn_m:.2f} → 합 {t_gpu_total:.2f} us")
print(f"  PIM decode-attn = {t_pim:.2f} us")
print(f"  → GPU/PIM = {t_gpu_total/t_pim:.1f}  (PIM 유휴율 ≈ {max(0,1-t_pim/t_gpu_total)*100:.1f}%)")
