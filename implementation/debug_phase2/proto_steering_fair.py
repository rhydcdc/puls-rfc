"""공정성(age-cap) 추가 steering 검증 — starvation 해소 + 균형 유지 확인.

pure closest-to-ideal 은 ideal-크기(~100K)만 cherry-pick → 짧/긴 요청 starvation(원소 분석으로 발견).
수정: 기본은 closest-to-ideal(균형), 단 wait ≥ AGE_CAP 인 요청은 강제 포함(공정성, FIFO 의도).
강제된 off-size 는 steering 이 보정(짧은 거 넣으면 ideal↑ → 다음에 긴 거) → 균형 유지.

스트리밍 시뮬(arrival mix 계속 유입)로 검증:
  (a) 서빙된 길이 분포 ≈ arrival 분포 (모든 클래스 drain, starvation 0)
  (b) 매 배치 (count≈123, Σkv≈12.3M) 균형
  (c) 최대 대기 ≤ AGE_CAP 배치 (starvation 상한)

실행: cd implementation && python debug_phase2/proto_steering_fair.py
"""
import sys
from pathlib import Path
import random
import collections

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puls_sched.config import default_dummy_config, compute_ffn_op_time_s
from puls_sched.micro_batch import MicroBatch
from puls_sched.pim_emulator import PIMExecutor

CFG = default_dummy_config()
PIM = PIMExecutor(CFG)
TC, TK, PF = 123, 12_300_000, 256
AGE_CAP = 2     # 요청은 최대 AGE_CAP 배치 대기 후 강제 포함


def t_pim(kv):
    return PIM.op_time(kv_rows_total=kv) * 1e-3 if kv > 0 else 0.0


def t_ffn(c):
    mb = MicroBatch(id=0, kv_rows_total=0, kv_rows_lockstep=0,
                    prefill_chunk={1: list(range(PF))}, decode_tokens={i: 0 for i in range(c)},
                    prefill_chunk_budget=PF, prefill_processed={1: 0})
    return compute_ffn_op_time_s(mb, CFG.calibration, CFG.model, num_gpus=8) * 1e6


def bucket(kv):
    return "<60K" if kv < 60_000 else "60-150K" if kv < 150_000 else "150-300K" if kv < 300_000 else ">300K"


def compose(pool):
    """pool = list of [kv, wait], arrival order. age-cap steering. picked 제거, 나머지 wait++."""
    n = S = 0
    picks, waits = [], []
    while n < TC and S < TK and pool:
        ideal = (TK - S) / (TC - n)
        aged = [i for i, r in enumerate(pool) if r[1] >= AGE_CAP]
        if aged:
            idx = aged[0]                                    # 가장 오래된 aged (arrival 순서 = 낮은 index)
        else:
            idx = min(range(len(pool)), key=lambda i: abs(pool[i][0] - ideal))
        r = pool.pop(idx)
        picks.append(r[0]); waits.append(r[1]); S += r[0]; n += 1
    for r in pool:
        r[1] += 1
    return picks, n, S, (max(waits) if waits else 0)


def gen_kv(rng):
    # 타깃 워크로드: 변종이되 *평균 ctx ≈ 100K* (동작점 도달 가능 영역). 짧+중+긴 섞임.
    r = rng.random()
    if r < 0.4:   return max(1000, int(rng.gauss(40_000, 15_000)))   # 짧음 40%
    if r < 0.7:   return max(1000, int(rng.gauss(100_000, 20_000)))  # 중간 30%
    return max(1000, int(rng.gauss(190_000, 30_000)))                # 긴 30% (avg ≈ 103K)


def run(n_batches=60, arrivals_per_batch=124):
    rng = random.Random(42)
    pool = []
    served = collections.Counter()
    arrived = collections.Counter()
    spreads, counts, sums, maxwaits = [], [], [], []
    for b in range(n_batches):
        for _ in range(arrivals_per_batch):
            kv = gen_kv(rng); pool.append([kv, 0]); arrived[bucket(kv)] += 1
        picks, n, S, mw = compose(pool)
        for k in picks:
            served[bucket(k)] += 1
        sp = abs(t_pim(S) - t_ffn(n)) / max(t_pim(S), t_ffn(n)) * 100
        spreads.append(sp); counts.append(n); sums.append(S); maxwaits.append(mw)

    warm = slice(10, None)  # warmup 배제
    print(f"AGE_CAP={AGE_CAP}, {n_batches} batches, arrivals {arrivals_per_batch}/batch\n")
    print("(a) 서빙 분포 vs arrival 분포 (모든 클래스 drain = starvation 0):")
    tot_s, tot_a = sum(served.values()), sum(arrived.values())
    for b in ["<60K", "60-150K", "150-300K", ">300K"]:
        print(f"    {b:9}: 서빙 {served[b]/tot_s*100:4.1f}%  vs  arrival {arrived[b]/tot_a*100:4.1f}%")
    print(f"\n(b) 균형 (warmup 배제 batch 10~): count avg {sum(counts[warm])/len(counts[warm]):.0f}, "
          f"Σkv avg {sum(sums[warm])/len(sums[warm])/1e6:.2f}M, spread avg {sum(spreads[warm])/len(spreads[warm]):.1f}%")
    print(f"(c) 최대 대기: {max(maxwaits[warm])} batch (상한 AGE_CAP={AGE_CAP})")
    print(f"\n원소 분석 — batch 20 의 버킷 구성 예시:")
    rng2 = random.Random(42); pool2 = []
    for b in range(21):
        for _ in range(arrivals_per_batch):
            pool2.append([gen_kv(rng2), 0])
        picks, n, S, mw = compose(pool2)
    bb = collections.Counter(bucket(k) for k in picks)
    print(f"    n={n} Σkv={S/1e6:.2f}M  버킷={dict(bb)}")


if __name__ == "__main__":
    run()
