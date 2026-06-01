"""로컬 그리디 steering 배치 구성 검증 (OPERATING_POINT.md former-v2 후보).

동작점은 count≈123 AND Σkv≈12.3M (둘 다 = avg ctx 100K) 을 동시에 요구한다. 순수 FIFO 는
Σkv 만 잡고 count 를 놓치므로, 전역 통계 없이 *로컬* 로 둘 다 맞추는 steering 을 검증:

  매 step: ideal = (target_kv − S) / (target_count − n)   # 남은합 ÷ 남은개수
           풀에서 ideal 에 가장 가까운(상한 안 넘는) 디코더 admit
  → (123, 12.3M) 에 자기보정 수렴. 미래예측·전역통계 없음. 단조라 ≤123 step 종료.

여러 분포(정규/heavy-tail/short-heavy/bimodal/uniform-long/uniform-short)에서 일반화 확인.
검증 = 각 배치의 t_PIM(Σkv) ≈ t_FFN(count+256) (둘 다 ~50.5µs면 균형). FIFO 와 대조.

실행: cd implementation && python debug_phase2/proto_steering.py
"""
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puls_sched.config import default_dummy_config, compute_ffn_op_time_s
from puls_sched.micro_batch import MicroBatch
from puls_sched.pim_emulator import PIMExecutor

CFG = default_dummy_config()
PIM = PIMExecutor(CFG)
TARGET_COUNT = 123
TARGET_KV = 12_300_000
UPPER_KV = 13_500_000     # +10% 오버슈트 천장
PREFILL = 256


def t_pim(kv):
    return PIM.op_time(kv_rows_total=kv) * 1e-3 if kv > 0 else 0.0


def t_ffn(count):
    mb = MicroBatch(id=0, kv_rows_total=0, kv_rows_lockstep=0,
                    prefill_chunk={1: list(range(PREFILL))},
                    decode_tokens={i: 0 for i in range(count)},
                    prefill_chunk_budget=PREFILL, prefill_processed={1: 0})
    return compute_ffn_op_time_s(mb, CFG.calibration, CFG.model, num_gpus=8) * 1e6


def steer_batch(pool, used):
    """로컬 그리디: 매 step 이상 길이에 가장 가까운(상한 안 넘는) 디코더 선택."""
    n, S, members = 0, 0, []
    while n < TARGET_COUNT and S < TARGET_KV:
        ideal = (TARGET_KV - S) / (TARGET_COUNT - n)
        best_i, best_d = -1, float("inf")
        for i, kv in enumerate(pool):
            if used[i] or S + kv > UPPER_KV:
                continue
            d = abs(kv - ideal)
            if d < best_d:
                best_d, best_i = d, i
        if best_i < 0:
            break
        used[best_i] = True
        S += pool[best_i]
        n += 1
        members.append(best_i)
    return n, S


def fifo_batch(pool, used):
    """대조군: 순수 FIFO, Σkv≥target 에서 stop (상한 skip)."""
    n, S = 0, 0
    for i, kv in enumerate(pool):
        if used[i]:
            continue
        if n >= TARGET_COUNT or S >= TARGET_KV:
            break
        if S + kv > UPPER_KV:
            continue
        used[i] = True
        S += kv
        n += 1
    return n, S


def evaluate(n, S):
    tp, tf = t_pim(S), t_ffn(n)
    spread = abs(tp - tf) / max(tp, tf) * 100 if max(tp, tf) > 0 else 0
    return tp, tf, spread


def run(pool, label):
    print(f"\n=== {label}  (pool {len(pool)}, mean {sum(pool)/len(pool)/1000:.0f}K) ===")
    for name, fn in [("steering", steer_batch), ("FIFO    ", fifo_batch)]:
        used = [False] * len(pool)
        line = []
        for _ in range(3):
            n, S = fn(pool, used)
            tp, tf, sp = evaluate(n, S)
            line.append(f"N{n:3d} Σ{S/1e6:4.1f}M PIM{tp:4.1f}/FFN{tf:4.1f} sp{sp:3.0f}%")
        print(f"  {name}: " + " | ".join(line))


def gen(kind, rng, n=3000):
    out = []
    for _ in range(n):
        if kind == "normal":
            out.append(max(1000, int(rng.gauss(100_000, 30_000))))
        elif kind == "heavy":
            r = rng.random()
            out.append(max(1000, int(rng.gauss(40_000, 15_000))) if r < 0.4
                       else max(1000, int(rng.gauss(100_000, 20_000))) if r < 0.7
                       else int(rng.uniform(200_000, 2_000_000)))
        elif kind == "short_heavy":
            out.append(max(1000, int(rng.gauss(30_000, 10_000))) if rng.random() < 0.7
                       else int(rng.uniform(120_000, 300_000)))
        elif kind == "bimodal":
            out.append(int(rng.gauss(30_000, 8_000)) if rng.random() < 0.5
                       else int(rng.gauss(200_000, 30_000)))
        elif kind == "uniform_long":
            out.append(int(rng.gauss(300_000, 40_000)))
        elif kind == "uniform_short":
            out.append(int(rng.gauss(40_000, 10_000)))
    return [max(1000, x) for x in out]


if __name__ == "__main__":
    print(f"target: count {TARGET_COUNT}, Σkv {TARGET_KV/1e6}M (avg {TARGET_KV/TARGET_COUNT/1000:.0f}K), "
          f"prefill {PREFILL}. 균형이면 PIM≈FFN≈50.5µs, sp≈0.")
    rng = random.Random(42)
    for kind, lbl in [
        ("normal", "정규 N(100K,30K) — 타깃"),
        ("heavy", "heavy-tail 혼합 (짧40/중30/긴30)"),
        ("short_heavy", "short-heavy (짧70/긴30)"),
        ("bimodal", "bimodal (30K 50% / 200K 50%)"),
        ("uniform_long", "uniform-long (~300K) — 불가능 예상"),
        ("uniform_short", "uniform-short (~40K) — 불가능 예상"),
    ]:
        run(gen(kind, rng), lbl)
