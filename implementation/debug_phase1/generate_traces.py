"""Phase-1 합성 트레이스 생성 — T-S / T-L / T-M (PLAN.md §3).

설계 근거 (코드 제약):
- CSV schema = trace.py TraceReplayer.load: arrived_at, num_prefill_tokens, num_decode_tokens
- kv_length = prefill + decode (trace.py replay). KV capacity 4,000,000 (config.py).
- 임계 컨텍스트 ≈ 56,160 (PIM/GPU = ctx/56160). ctx > 56K → PIM bound.
- decode = max_tokens (completion.py). decode 분산 = decode 단계 체류 분산.
- 도착 포화 — idle 분모 = wall-clock span (idle_telemetry.py). inter-arrival 조밀.

세 트레이스의 검증 명제:
- T-S (short 몰림, ctx ≪ 56K) — PIM idle 큼 (정상). 음성 대조군.
- T-L (long 몰림, ctx ≫ 56K) — PIM bound, GPU 유휴. chunked prefill 합류 버그 노출.
- T-M (혼합 short:long ≈ 3:7) — 시점별 바인딩 반전. 양방향 balance 필요.

소스 무수정 — 별도 CSV 산출 후 Run 에 path 주입.
"""

import argparse
import csv
import math
import random
from pathlib import Path

CTX_THRESHOLD = 56_160          # PIM/GPU 균형 컨텍스트 (분석 §1)
KV_CAPACITY = 4_000_000         # config.py — 동시 admit kv_length 합 상한


def _clamped_lognormal(rng, mu, sigma, lo, hi):
    """lognormal 표본을 [lo, hi] 로 clamp 한 정수."""
    return int(min(hi, max(lo, rng.lognormvariate(mu, sigma))))


def _decode_tokens(rng):
    """decode 토큰 수 — 고분산 (도구호출 짧음 ~ 긴 추론). 실 트레이스 350 고정 탈피."""
    # median e^6.0 ≈ 403, 꼬리 두껍게 (sigma 0.8) → 대략 [50, 2000]
    return _clamped_lognormal(rng, mu=6.0, sigma=0.8, lo=50, hi=2000)


def _arrivals(rng, n, mean_interval):
    """누적 exponential 도착. mean_interval 작을수록 포화."""
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_interval)
        out.append(t)
    return out


def gen_short(rng, n):
    """T-S — prefill 전부 ctx ≪ 56K. median e^9.0 ≈ 8103, clamp [1000, 40000]."""
    rows = []
    for at in _arrivals(rng, n, mean_interval=0.5):
        pre = _clamped_lognormal(rng, mu=9.0, sigma=0.6, lo=1_000, hi=40_000)
        rows.append((at, pre, _decode_tokens(rng)))
    return rows


def gen_long(rng, n):
    """T-L — prefill ctx 56K–150K. KV캐파 내 동시 수십 개 생존하도록 대역 제한."""
    rows = []
    for at in _arrivals(rng, n, mean_interval=0.5):
        pre = rng.randint(58_000, 150_000)
        rows.append((at, pre, _decode_tokens(rng)))
    return rows


def gen_mixed(rng, n):
    """T-M — short:long ≈ 3:7. long-context 우세 (agentic 누적 이력 정합).

    prefill 분산을 실 트레이스급으로 두껍게 — long 표본 중 일부는 heavy tail (수십만).
    """
    rows = []
    for at in _arrivals(rng, n, mean_interval=0.5):
        if rng.random() < 0.3:
            pre = _clamped_lognormal(rng, mu=9.0, sigma=0.6, lo=1_000, hi=40_000)
        else:
            # long — 대부분 60K–150K, 10% heavy tail 300K–800K
            if rng.random() < 0.1:
                pre = rng.randint(300_000, 800_000)
            else:
                pre = rng.randint(58_000, 150_000)
        rows.append((at, pre, _decode_tokens(rng)))
    return rows


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        for at, pre, dec in rows:
            w.writerow([at, pre, dec])


def _stats(name, rows):
    pre = [r[1] for r in rows]
    dec = [r[2] for r in rows]
    kv = [r[1] + r[2] for r in rows]
    n = len(rows)
    over = sum(1 for c in pre if c > CTX_THRESHOLD)
    mean_kv = sum(kv) / n
    est_concurrency = KV_CAPACITY / mean_kv
    print(f"[{name}] n={n}")
    print(f"  prefill  min/mean/max = {min(pre)}/{int(sum(pre)/n)}/{max(pre)}")
    print(f"  decode   min/mean/max = {min(dec)}/{int(sum(dec)/n)}/{max(dec)}")
    print(f"  ctx>56K  = {over}/{n} ({100*over/n:.0f}%)")
    print(f"  mean_kv  = {int(mean_kv)}  -> est. concurrency = {est_concurrency:.1f} (KVcap/mean_kv)")
    print()


def main():
    ap = argparse.ArgumentParser(description="Phase-1 합성 트레이스 생성")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--n-short", type=int, default=40)
    ap.add_argument("--n-long", type=int, default=20)
    ap.add_argument("--n-mixed", type=int, default=40)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "data"))
    ns = ap.parse_args()

    out = Path(ns.out)
    specs = [
        ("T-S", "trace_short.csv", gen_short, ns.n_short),
        ("T-L", "trace_long.csv", gen_long, ns.n_long),
        ("T-M", "trace_mixed.csv", gen_mixed, ns.n_mixed),
    ]
    for name, fname, fn, n in specs:
        rng = random.Random(ns.seed + hash(name) % 1000)
        rows = fn(rng, n)
        _write(out / fname, rows)
        _stats(name, rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
