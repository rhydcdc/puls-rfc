"""클러스터 레이어: 노드 풀 100K 센터링 라우팅 (ARCHITECTURE §7) — standalone 레퍼런스.

이 모듈은 **의도를 기록**하기 위한 독립 구현이다. `puls_sched` 패키지(노드 내부
스케줄러)와 연결되어 동작하지 않으며, 어느 모듈도 이걸 import 하지 않는다. 검증은
C++ sim [implementation/analysis/cluster_balance.cpp] 에서 했고(ARCH §7.5), 여기선
같은 로직을 읽기 좋은 파이썬으로 옮겨 둔다.

배경(ARCH §6.4, §7): PULS 동작점은 한 노드의 in-flight 풀이 ~100K 로 센터된 변종
풀일 때 steering 이 명중한다(개수 123 ∧ Σkv 12.3M). 서버스케일에선 글로벌 도착
평균이 100K 보다 높아 노드별 풀이 drift → count/Σkv 동시 이탈 → idle 폭발. 그래서
클러스터 레이어가 (1) 과도하게 긴 요청을 엣지로 격리하고, (2) 정상 노드를 평균 100K
로 채우며(cold-start), (3) 완료로 빈 자리를 풀에서 보충해 동작점을 유지한다(healing).
inter-node swap 0, 무축출(완료 순간만 churn).

avg 100K 는 워크로드에 강제하는 값이 아니라 30M 캡에 246~300 디코더를 fit 시키는
*용량/개수* 조건이다(§6.4). 123-배치의 12.3M 명중은 steering composer + 풀 다양성의
몫이며, 노드 300-평균이 그걸 직접 보장하진 않는다(§7.2).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ── 동작점 상수 (ARCH §6.4 / OPERATING_POINT.md) ──────────────────────────────
Z = 256                      # 정상(PULS) 노드 수
CAP = 30_000_000             # 노드당 footprint 캡 (토큰)
TGT_AVG = 100_000.0          # 노드 상주 풀 센터링 타깃 (12.3M / 123, 캡 유도용 중간값)
TGT_CNT = 123                # decode 개수 타깃
TGT_KV = 12_300_000          # decode-KV 합 타깃
NODE_MAX = 300               # CAP / TGT_AVG = 30M / 100K
NODE_MIN = 246               # 2 μ-batch floor (= 123 × 2)
EDGE_BAND = 1_000            # cold-start 게이팅 밴드 E (ARCH §7.5: E=1K 채택)


# ── 가정 분포 B (실 트래픽 부재; ARCH §7) ─────────────────────────────────────
# short 20% [1-16K] / mid 70% [16-256K] / long 10% [256K-1M], 평균 ≈ 116K.
def sample_b(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.20:
        lo, hi = 1_000, 16_000
    elif u < 0.90:
        lo, hi = 16_000, 256_000
    else:
        lo, hi = 256_000, 1_000_000
    return int(math.exp(rng.random() * (math.log(hi) - math.log(lo)) + math.log(lo)))


@dataclass
class Node:
    """한 PULS 노드의 상주 디코더 풀 (길이 = KV footprint 근사)."""
    lengths: list[int] = field(default_factory=list)
    total: int = 0                                       # Σ length (= 상주 KV 합)

    @property
    def count(self) -> int:
        return len(self.lengths)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.lengths else 0.0

    def admit(self, length: int) -> None:
        self.lengths.append(length)
        self.total += length

    def can_fit(self, length: int) -> bool:
        return self.total + length <= CAP and self.count < NODE_MAX


# ── Step 1: 게이팅 (엣지 격리) ────────────────────────────────────────────────
def gate(draw: list[int], band: int = EDGE_BAND) -> tuple[list[int], list[int]]:
    """긴 것부터 떼어 남은 평균이 100K+band 이하가 될 때까지 엣지로 격리.

    edge 비율 = f(band) 단독 (분포 B 의 임계 위 꼬리 P(x>V)) — 노드 수·풀 크기 무관.
    반환: (kept, edged).
    """
    ordered = sorted(draw, reverse=True)                 # desc: 가장 긴 것부터
    total = sum(ordered)
    ceil = TGT_AVG + band
    i = 0
    n = len(ordered)
    while n > 0 and total / n > ceil:                    # 평균이 천장 위면 맨 앞(최장) shed
        total -= ordered[i]
        i += 1
        n -= 1
    return ordered[i:], ordered[:i]                      # kept(짧→긴 순), edged


# ── Step 2: cold-start — interleave greedy ───────────────────────────────────
def cold_start(kept: list[int], rng: random.Random,
               nodes: list[Node] | None = None) -> tuple[list[Node], int]:
    """남은 요청을 도착 순(shuffle)으로 min|추가후 mean - 100K| 노드에 적재.

    긴 거+짧은 거가 한 노드에 자연 interleave → count 300 · cap 30M · 평균 100K 동시 충족.
    (편차 크기순 정렬은 cap 을 적은 count 로 먼저 채워 실패 — ARCH §7.3. 그래서 interleave.)
    배치 못한 요청 수(leftover)는 edge 로 합산해 반환.
    """
    if nodes is None:
        nodes = [Node() for _ in range(Z)]
    order = kept[:]
    rng.shuffle(order)
    leftover = 0
    for length in order:
        best: Node | None = None
        best_dev = math.inf
        for nd in nodes:
            if not nd.can_fit(length):
                continue
            dev = abs((nd.total + length) / (nd.count + 1) - TGT_AVG)
            if dev < best_dev:
                best_dev, best = dev, nd
        if best is None:                                 # 전 노드 만석/캡 → 못 받음
            leftover += 1
            continue
        best.admit(length)
    return nodes, leftover


# ── Step 3: healing — 전략적 greedy refill (무축출) ──────────────────────────
def _pull_best(ideal: float, cap_room: int, rng: random.Random, k: int = 200) -> int | None:
    """무한 풀(분포 B)에서 ideal 에 가장 가까운(캡 적합) 요청을 best-of-k 로. (풀 일부만 사용 = 무한 stock)"""
    best: int | None = None
    best_dev = math.inf
    for _ in range(k):
        length = sample_b(rng)
        if length > cap_room:
            continue
        dev = abs(length - ideal)
        if dev < best_dev:
            best_dev, best = dev, length
    return best


def heal(node: Node, departed: list[int], rng: random.Random) -> int:
    """per-completion 힐링: 완료로 빠진 각 hole 을 그 크기(ideal=hole)로 되채움.

    실서버 admission(§6.4)은 완료 1개마다 즉시 1개 admit(backpressure). 그러니 slot=1,
    ideal = (목표 footprint - sum)/1 = hole(빠진 그 크기) → 풀에서 ≈hole 을 당김 =
    like-for-like. **긴 게 빠지면 긴 게 들어온다 = Phase-1 toxic-fit**(ARCH §7.4).

    효과: 상주 길이분포 보존(긴 요청 안 굶음, 측정 8.15%→7.34%) · 각 클래스 도착률대로
    소비 → 엣지 cold-start 비율 유지. inter-node swap 없음.

    ⚠ 여러 완료를 모아 ideal=평균 으로 한꺼번에 채우면(batched) toxic-fit 깨져 긴 게
    0% 로 굶는다(§7.4). 그래서 반드시 완료-단위. pulls 수 반환.
    """
    pulls = 0
    for hole in departed:
        cap_room = CAP - node.total
        if cap_room <= 0:
            break
        length = _pull_best(float(hole), cap_room, rng)   # ideal = hole(빠진 크기)
        if length is None:
            break
        node.admit(length)
        pulls += 1
    return pulls


# ── on-point 체크 (검증용) — 노드 풀에서 (123, 12.3M±10%) 배치 K개 뽑히나 ────────
def disjoint_onpoint_batches(node: Node, k: int = 2) -> int:
    """steering(closest-to-ideal)으로 disjoint 배치를 최대 k개 compose. 명중 개수 반환.

    on2(=k=2)가 floor 246 의 진짜 의미: 2 μ-batch 각각 (123, 12.3M) 명중 가능한가(ARCH §7.2).
    """
    pool = sorted(node.lengths)
    used = [False] * len(pool)
    got = 0
    for _ in range(k):
        s = 0
        n = 0
        picks: list[int] = []
        while n < TGT_CNT and s < TGT_KV:
            ideal = round((TGT_KV - s) / (TGT_CNT - n))
            best_i = -1
            best_dev = math.inf
            for i, length in enumerate(pool):            # closest-to-ideal 미사용 원소
                if used[i]:
                    continue
                dev = abs(length - ideal)
                if dev < best_dev:
                    best_dev, best_i = dev, i
            if best_i < 0:
                break
            used[best_i] = True
            picks.append(best_i)
            s += pool[best_i]
            n += 1
        if n == TGT_CNT and abs(s - TGT_KV) / TGT_KV <= 0.10:
            got += 1
        else:
            for i in picks:                              # 실패 배치 롤백·중단
                used[i] = False
            break
    return got


# ── 데모: cold-start → healing 안정성 (ARCH §7.5 축약 재현) ───────────────────
def _demo() -> None:
    rng = random.Random(12345)
    # draw ≈ 용량(Z×NODE_MAX) — kept 가 클러스터를 거의 채우게(잉여 stock 최소).
    draw = [sample_b(rng) for _ in range(79_000)]
    kept, edged = gate(draw, EDGE_BAND)
    nodes, leftover = cold_start(kept, rng)

    # edge% = 게이트가 떼낸 비율 = 분포 꼬리 P(x>V) (ARCH §7.3, 스케일 불변).
    # leftover(배치 못한 surplus)는 무한 풀 stock 이지 edge 아님.
    in_range = sum(NODE_MIN <= nd.count <= NODE_MAX for nd in nodes)
    on2 = sum(disjoint_onpoint_batches(nd, 2) >= 2 for nd in nodes)
    print("[cold-start]  E=%dK  edge=%.2f%%  count∈246-300=%.0f%%  on2=%.0f%%  (leftover stock=%d)" % (
        EDGE_BAND // 1000, 100 * len(edged) / len(draw),
        100 * in_range / Z, 100 * on2 / Z, leftover))

    long_cold = sum(L >= 256_000 for nd in nodes for L in nd.lengths)
    long_tot = sum(nd.count for nd in nodes)
    p = 0.03                                             # 라운드당 완료확률
    for rd in range(200):
        for nd in nodes:                                 # churn: 완료분 retire → per-completion heal
            keep, departed = [], []
            for L in nd.lengths:
                (keep if rng.random() >= p else departed).append(L)
            nd.lengths, nd.total = keep, sum(keep)
            heal(nd, departed, rng)                       # 각 hole 을 그 크기로 되채움(toxic-fit)
    means = [nd.mean for nd in nodes]
    on2 = sum(disjoint_onpoint_batches(nd, 2) >= 2 for nd in nodes)
    long_late = sum(L >= 256_000 for nd in nodes for L in nd.lengths)
    long_tot2 = sum(nd.count for nd in nodes)
    print("[healing 200rd]  count∈246-300=%.0f%%  |mean-100K|avg=%.0f토큰  on2=%.0f%%" % (
        100 * sum(NODE_MIN <= nd.count <= NODE_MAX for nd in nodes) / Z,
        sum(abs(m - TGT_AVG) for m in means) / Z, 100 * on2 / Z))
    print("[toxic-fit]  상주 긴요청(≥256K)  cold=%.2f%% → late=%.2f%%  (보존되면 긴 게 안 굶음)" % (
        100 * long_cold / long_tot, 100 * long_late / long_tot2))


if __name__ == "__main__":
    _demo()
