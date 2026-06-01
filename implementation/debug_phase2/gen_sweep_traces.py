"""S4(d) 스케일 스펙트럼 스윕용 트레이스 생성 (사용자 케이스 분할 2026-06-02).

per-cycle 균형의 주 knob = **prefill 길이(ctx)** — GPU-A prefill-attn depth *와* 디코더 ctx
(PIM decode KV) 둘 다 결정. decode max_tokens 는 주로 수명/throughput. uniform = "고른 분포".

  A_even      : prefill 20K~180K 고름(sweet spot ~100K 포함), decode 100~600 고름 — 실서버형.
                steering 이 변종 풀서 조합해 타깃(prefill depth-work 25.6M, decode Σkv 12.3M)
                명중 가능 → idle≈0 *저절로* 나오는지 = 알고리즘 검증 핵심.
  B_pflong    : prefill 100K~250K(깊음), decode 100~600 고름 — longbench류 → GPU-A bound 경향.
  C_both_long : prefill 100K~250K, decode 400~600(긺) → Instance-A bound 강함.

prefill 짧고 decode 긴 케이스는 비현실(에이전틱도 prefill 이 더 김)이라 제외(사용자).
출력: data/sweep_{A,B,C}.csv (schema = longbench 동일: arrived_at,num_prefill_tokens,num_decode_tokens)
"""
import random
from pathlib import Path

N = 5000
RATE = 0.05         # arrival rate (seed-기반 측정이라 timing 영향 작음 — 보충용)
CASES = {
    "sweep_A_even":      dict(pf=(20_000, 180_000), dc=(100, 600)),
    "sweep_B_pflong":    dict(pf=(100_000, 250_000), dc=(100, 600)),
    "sweep_C_both_long": dict(pf=(100_000, 250_000), dc=(400, 600)),
    # D = 실서버형: prefill 다양(sweet spot 포함) + **decode 긴**(대량 상주 decode 풀).
    # idle≈0 발현 검증의 주 워크로드(README Runtime Validation) — 풀이 풍부해 동작점 명중.
    "sweep_D_longdec":   dict(pf=(20_000, 180_000), dc=(8_000, 40_000)),
}


def gen(name, pf, dc):
    rng = random.Random(42)
    t = 0.0
    path = Path("data") / f"{name}.csv"
    with open(path, "w", newline="") as f:
        f.write("arrived_at,num_prefill_tokens,num_decode_tokens\n")
        for _ in range(N):
            t += rng.expovariate(RATE)
            f.write(f"{t},{rng.randint(*pf)},{rng.randint(*dc)}\n")
    print(f"{path}  n={N} prefill={pf} decode={dc}")


if __name__ == "__main__":
    for name, cfg in CASES.items():
        gen(name, cfg["pf"], cfg["dc"])
