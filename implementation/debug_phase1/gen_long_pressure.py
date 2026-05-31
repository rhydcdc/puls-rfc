"""확장 T-L — KV 캐파(4M) 압박용. 합류 부재 확증 baseline.

목적: 동시 생존 in-flight KV 합이 4M 을 넘도록 충분한 요청을 촘촘한 도착으로 투입.
캐파 초과분이 큐에 적체 → 새 mb 다중 생성(mb 다중화는 됨) → 그러나 기존 mb 에
합류는 0 (admission.layer1 이 in_flight 미관측). cross-mb staggering 만 발생.

설계 (코드 근거):
- kv_length = prefill + decode. 평균 ~100K → 동시 ~40 요청에서 4M 천장.
  천장 초과를 강제하려면 동시 생존이 40 을 넘게 도착을 촘촘히.
- decode 짧게(24–64) — 런타임 절약. 단 너무 짧으면 빨리 완료→release→압박 약화.
  도착 간격을 release 보다 작게 하여 in_flight 누적.
- 전부 ctx > 56K (long regime).
"""
import csv
import random
from pathlib import Path

SEED = 20260531
N = 80                      # 동시 ~40 천장의 2배 — 적체 보장
rng = random.Random(SEED)

rows = []
t = 0.0
for i in range(N):
    # 촘촘한 도착 — 평균 간격 작게 (release 보다 빠르게 누적)
    t += rng.expovariate(1.0 / 0.15)
    prefill = rng.randint(58_000, 150_000)     # ctx > 56K
    decode = rng.randint(24, 64)               # 짧게 (런타임 절약)
    rows.append((t, prefill, decode))

out = Path(__file__).parent / "data" / "trace_long_pressure.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
    for at, pre, dec in rows:
        w.writerow([at, pre, dec])

kv = [r[1] + r[2] for r in rows]
print(f"N={N}  out={out}")
print(f"  prefill min/mean/max = {min(r[1] for r in rows)}/{int(sum(r[1] for r in rows)/N)}/{max(r[1] for r in rows)}")
print(f"  decode  min/mean/max = {min(r[2] for r in rows)}/{int(sum(r[2] for r in rows)/N)}/{max(r[2] for r in rows)}")
print(f"  kv mean = {int(sum(kv)/N)}  -> 4M / mean = 동시 천장 {4_000_000/(sum(kv)/N):.1f} 요청")
print(f"  마지막 도착 ts = {rows[-1][0]:.2f}")
print(f"  Sigma kv (전체 누적) = {sum(kv)}  (>4M? {sum(kv) > 4_000_000})")
