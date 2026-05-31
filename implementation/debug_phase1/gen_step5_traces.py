"""STEP 5 재검증용 트레이스 생성 — T-DEC / T-M / T-S (PLAN STEP 5 재설계).

구 T-L/long_pressure 는 prefill 지배라 (a) prefill 구간엔 decode 일감 부재 → 합류가
PIM 못 채움, (b) 측정이 prefill 구간에 갇힘. 합류 효과의 핵심 = **decode-bound 정상
상태에서 prefill backlog 로 GPU 를 채움**. 그에 맞춰 재설계.

세 트레이스 (역할 분담):
- T-DEC (주 demonstrator) — ctx 58–72K(임계 56,160 바로 위 → decode PIM-bound, prefill
  짧음), decode 고분산(prefill cycle 지배 → run 이 decode-bound), N≈80(ΣKV>4M → prefill
  backlog 상존). 합류 ON 시 backlog prefill 이 GPU 채워 GPU idle 급감 기대.
- T-M (현실성) — 혼합 short+long · decode 분산. PULS 타깃(agentic·멀티턴·KV 길이 분산)
  에 가장 근접. 양방향(PIM·GPU 동시 채움) 확인.
- T-S (대조군) — short(ctx ≪ 56K), idle 불변 확인 (과잉 수정 방지).

소스 무수정 — 별도 CSV 산출 후 measure_steady 에 path 주입. 기존 trace_*.csv 와
파일명 분리 (trace_tdec/tm/ts.csv).
"""
import argparse
import csv
import random
from pathlib import Path

CTX_THRESHOLD = 56_160      # PIM/GPU 균형 컨텍스트
KV_CAPACITY = 4_000_000     # config.py — 동시 admit kv_length 합 상한


def _clamped_lognormal(rng, mu, sigma, lo, hi):
    return int(min(hi, max(lo, rng.lognormvariate(mu, sigma))))


def _arrivals(rng, n, mean_interval):
    """누적 exponential 도착. mean_interval 작을수록 포화(backlog 형성)."""
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_interval)
        out.append(t)
    return out


def gen_tdec(rng, n):
    """T-DEC — long-context agentic: prompt 크고 decode 가 따라잡음(~prompt 절반).

    causal 비대칭(prefill-attn 은 각 토큰이 앞부분만 = 평균 절반 ctx, decode-attn 은
    각 토큰이 full ctx)으로 **decode ≈ prompt/2 면 decode-attn 일이 prefill-attn 에
    맞먹거나 넘어섬** → PIM 활용. 실제 agentic 추론(긴 CoT)에 정합.

    가벼운 가늠 버전: prompt ~20K, decode ~10K (median). 좋으면 40K/20K 로 스케일.
    """
    rows = []
    for at in _arrivals(rng, n, mean_interval=0.5):
        pre = rng.randint(16_000, 24_000)
        dec = _clamped_lognormal(rng, mu=9.2, sigma=0.3, lo=6_000, hi=16_000)  # median ~9900
        rows.append((at, pre, dec))
    return rows


def gen_tm(rng, n):
    """T-M — short:long ≈ 3:7, decode 분산 큼(agentic: 짧은 tool-call ~ 긴 추론)."""
    rows = []
    for at in _arrivals(rng, n, mean_interval=0.5):
        if rng.random() < 0.3:
            pre = _clamped_lognormal(rng, mu=9.0, sigma=0.6, lo=1_000, hi=40_000)  # short
        elif rng.random() < 0.1:
            pre = rng.randint(300_000, 800_000)                                     # long heavy-tail
        else:
            pre = rng.randint(58_000, 150_000)                                      # long
        dec = _clamped_lognormal(rng, mu=6.2, sigma=0.7, lo=200, hi=2_000)
        rows.append((at, pre, dec))
    return rows


def gen_ts(rng, n):
    """T-S — short 몰림(ctx ≪ 56K). 대조군. decode 고분산."""
    rows = []
    for at in _arrivals(rng, n, mean_interval=0.5):
        pre = _clamped_lognormal(rng, mu=9.0, sigma=0.6, lo=1_000, hi=40_000)
        dec = _clamped_lognormal(rng, mu=6.0, sigma=0.8, lo=50, hi=1_226)
        rows.append((at, pre, dec))
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
    sum_kv = sum(kv)
    print(f"[{name}] n={n}")
    print(f"  prefill min/mean/max = {min(pre)}/{int(sum(pre)/n)}/{max(pre)}")
    print(f"  decode  min/mean/max = {min(dec)}/{int(sum(dec)/n)}/{max(dec)}")
    print(f"  ctx>56K = {over}/{n} ({100*over/n:.0f}%)")
    print(f"  ΣKV = {sum_kv:,} (cap {KV_CAPACITY:,}) → backlog? {'YES' if sum_kv > KV_CAPACITY else 'no'}"
          f" (ΣKV/cap = {sum_kv/KV_CAPACITY:.2f}×)")
    print()


def main():
    ap = argparse.ArgumentParser(description="STEP 5 재검증 트레이스 생성 (T-DEC/T-M/T-S)")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--n-tdec", type=int, default=80)   # ΣKV>4M backlog
    ap.add_argument("--n-tm", type=int, default=80)     # ΣKV>4M backlog
    ap.add_argument("--n-ts", type=int, default=40)     # 대조군, backlog 불필요
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "data"))
    ns = ap.parse_args()

    out = Path(ns.out)
    specs = [
        ("T-DEC", "trace_tdec.csv", gen_tdec, ns.n_tdec),
        ("T-M", "trace_tm.csv", gen_tm, ns.n_tm),
        ("T-S", "trace_ts.csv", gen_ts, ns.n_ts),
    ]
    for name, fname, fn, n in specs:
        rng = random.Random(ns.seed + hash(name) % 1000)
        rows = fn(rng, n)
        _write(out / fname, rows)
        _stats(name, rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
