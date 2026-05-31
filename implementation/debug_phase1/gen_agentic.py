"""현실적 agentic 장기추론 트레이스 — ctx>56K(긴 컨텍스트) + 긴 decode.

PULS balance/PIM 활용이 발현하는 regime: prompt>56K(ctx>56K → decode 가 PIM-bound)
+ 긴 decode(decode 구간이 lifecycle 지배 → PIM 바쁨). balance 가 prefill chunk 를
PIM 슬랙에 맞춰 신규 prefill 로 GPU 채움 → 둘 다 활용.

비싼 확인용 — decode 1만~3.5만 토큰 × 80층이라 무거움(수렴-정지로 측정 bound).
N 작게(30) 하여 cold-start prefill 가속(요청당 청크 큼) + warmup 단축.
"""
import csv
import random
from pathlib import Path

N = 30
SEED = 20260601


def _clamped_lognormal(rng, mu, sigma, lo, hi):
    return int(min(hi, max(lo, rng.lognormvariate(mu, sigma))))


def main():
    rng = random.Random(SEED)
    rows = []
    t = 0.0
    for _ in range(N):
        t += rng.expovariate(1.0 / 0.5)            # 포화 도착
        pre = rng.randint(58_000, 78_000)          # ctx>56K (긴 컨텍스트 읽기)
        dec = _clamped_lognormal(rng, mu=9.8, sigma=0.35, lo=10_000, hi=35_000)  # 긴 추론 (median ~18K)
        rows.append((t, pre, dec))

    out = Path(__file__).parent / "data" / "trace_agentic.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        for at, pre, dec in rows:
            w.writerow([at, pre, dec])

    pre = [r[1] for r in rows]
    dec = [r[2] for r in rows]
    kv = [r[1] + r[2] for r in rows]
    sum_kv = sum(kv)
    print(f"[agentic] n={N}")
    print(f"  prefill min/mean/max = {min(pre)}/{int(sum(pre)/N)}/{max(pre)}")
    print(f"  decode  min/mean/max = {min(dec)}/{int(sum(dec)/N)}/{max(dec)}")
    print(f"  ctx>56K = {sum(1 for c in pre if c > 56160)}/{N}")
    print(f"  decode/prompt 평균 비 = {sum(dec)/sum(pre):.2f}")
    print(f"  ΣKV = {sum_kv:,} (cap 4,000,000) → backlog? {'YES' if sum_kv > 4_000_000 else 'mild'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
