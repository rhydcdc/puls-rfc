"""일반/적당 스케일 트레이스 — 전형적 챗·경량 에이전트 (prompt·decode 둘 다 보통).

스케일 스펙트럼 검증용: agentic(고ctx 장기추론) 외에 보통 규모에서도 스케줄러가
건전히 동작하는지(과포화 없음·다중 mb·합리적 idle profile). ctx<56K → GPU-bound 가
정상(PIM 싸고 prefill-attn 작음).
"""
import csv
import random
from pathlib import Path

N = 50
SEED = 20260601


def _clamped_lognormal(rng, mu, sigma, lo, hi):
    return int(min(hi, max(lo, rng.lognormvariate(mu, sigma))))


def main():
    rng = random.Random(SEED)
    rows = []
    t = 0.0
    for _ in range(N):
        t += rng.expovariate(1.0 / 0.5)                                       # 포화 도착
        pre = _clamped_lognormal(rng, mu=9.4, sigma=0.4, lo=4_000, hi=20_000)  # 전형 prompt (median ~12K)
        dec = _clamped_lognormal(rng, mu=7.6, sigma=0.5, lo=300, hi=6_000)     # 전형 decode (median ~2K)
        rows.append((t, pre, dec))

    out = Path(__file__).parent / "data" / "trace_general.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        for at, pre, dec in rows:
            w.writerow([at, pre, dec])

    pre = [r[1] for r in rows]
    dec = [r[2] for r in rows]
    sum_kv = sum(p + d for p, d in zip(pre, dec))
    print(f"[general] n={N}")
    print(f"  prefill min/mean/max = {min(pre)}/{int(sum(pre)/N)}/{max(pre)}")
    print(f"  decode  min/mean/max = {min(dec)}/{int(sum(dec)/N)}/{max(dec)}")
    print(f"  ctx>56K = {sum(1 for c in pre if c > 56160)}/{N}")
    print(f"  decode/prompt 평균 비 = {sum(dec)/sum(pre):.2f}")
    print(f"  ΣKV = {sum_kv:,} (cap 4,000,000) → {'backlog' if sum_kv > 4_000_000 else 'mild'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
