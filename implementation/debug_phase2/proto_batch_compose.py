"""former 배치 구성 알고리즘 프로토타입 검증 (OPERATING_POINT.md §3).

FIFO + skip + in-band stop 만 떼어내, 정규분포 트레이스로 μ-batch 3개를 구성해
모두 decode-KV 밴드 [11.1M, 13.5M] 안에 드는지 확인. (스케줄러 구현 전 알고리즘 sanity.)

prefill 256 동작점: target 12.3M, 밴드 ±10% [11.1M, 13.5M], N_dec 천장 123.
실행: python debug_phase2/proto_batch_compose.py
"""
import random

LOWER, UPPER, TARGET, NDEC_CAP = 11.1e6, 13.5e6, 12.3e6, 123


def compose_batch(pool, used, stop_at):
    """단일 FIFO 패스(재스캔 없음 → 무한 hunt 불가). 도착순 누적, Σkv≥stop_at 도달 또는
    N_dec 천장 또는 패스 끝나면 마감. 오버슈트(>UPPER)는 skip(used 안 함 → 다음 batch 후보)."""
    total = 0.0
    members = []
    skipped = 0
    for i, kv in enumerate(pool):
        if used[i]:
            continue
        if len(members) >= NDEC_CAP:      # FFN 개수 천장
            break
        if total >= stop_at:              # 정지 임계 도달
            break
        if total + kv > UPPER:            # 오버슈트 → skip(defer to 다음 batch)
            skipped += 1
            continue
        used[i] = True
        total += kv
        members.append(i)
    return members, total, skipped


def run(pool, label, stop_at, stop_name):
    used = [False] * len(pool)
    print(f"\n=== {label} | stop={stop_name} (pool {len(pool)}, mean ctx {sum(pool)/len(pool)/1000:.0f}K) ===")
    all_members = []
    for b in range(3):
        members, total, skipped = compose_batch(pool, used, stop_at)
        inband = LOWER <= total <= UPPER
        avg = total / max(len(members), 1) / 1000
        idle = abs(total - TARGET) / TARGET * 100  # 중심 대비 편차 ≈ idle 상한
        print(f"  batch{b+1}: N_dec={len(members):3d}  Σkv={total/1e6:5.2f}M  avg={avg:3.0f}K  "
              f"skip={skipped:2d}  in-band={inband}  (target편차 {idle:.1f}%)")
        all_members.append(set(members))
    disjoint = (all_members[0].isdisjoint(all_members[1]) and
                all_members[1].isdisjoint(all_members[2]) and
                all_members[0].isdisjoint(all_members[2]))
    print(f"  → disjoint(3배치 중복 0): {disjoint}")


rng = random.Random(42)
# A — 정규분포 (대수의 법칙, 실 트래픽 근사)
pool_a = [max(1000, int(rng.gauss(100_000, 30_000))) for _ in range(2000)]
# B — heavy-tail 혼합 (longbench 류) — skip 동작 확인용
pool_b = []
for _ in range(2000):
    r = rng.random()
    if r < 0.4:   pool_b.append(max(1000, int(rng.gauss(40_000, 15_000))))   # 짧음 40%
    elif r < 0.7: pool_b.append(max(1000, int(rng.gauss(100_000, 20_000))))  # 중간 30%
    else:         pool_b.append(int(rng.uniform(200_000, 5_000_000)))        # 긴 tail 30%
random.Random(7).shuffle(pool_b)

# 두 stop 규칙 비교: 하한 진입 즉시(LOWER) vs 중심까지(TARGET). 둘 다 single-pass(무한 hunt X).
for pool, label in [(pool_a, "A. 정규분포 N(100K,30K)"), (pool_b, "B. heavy-tail 혼합")]:
    run(pool, label, LOWER, "LOWER(밴드 진입)")
    run(pool, label, TARGET, "TARGET(중심)")
