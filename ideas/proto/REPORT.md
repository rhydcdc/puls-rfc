# 스케줄러 대조 측정 보고 (A=baseline vs B=prepo)

**1줄 요약:** 글로벌 사전배치(B)는 A 대비 **p99 tail을 약 5.8× 단축**(12.67M→2.17M us)하고 **멀티턴 캐시 히트(hbmHit 82%)** 와 **공정성(forced·max_wait 유한)** 을 확보하지만, 노드 re-selection 제거의 대가로 **Σdev(idle 평균)는 오히려 악화**(1.64%→8.32%)한다. 이득의 본질은 평균 idle이 아니라 tail/cache/공정성이다. (B 수치는 재튜닝된 global_age_cap=1000 기준 — 자세한 재튜닝 근거는 REPORT2.md.)

---

## 빌드 / 실행

빌드 (repo 루트에서):
```
g++ -std=c++17 -O2 -Wall -Wextra -Iideas/proto ideas/proto/sim/baseline.cpp ideas/proto/scheduler/queue.cpp ideas/proto/scheduler/cache.cpp ideas/proto/sim/workload_mt.cpp ideas/proto/core/derive.cpp ideas/proto/core/optime.cpp ideas/proto/core/steering.cpp ideas/proto/core/global_scheduler.cpp ideas/proto/core/workload.cpp -o ideas/proto/build/baseline.exe

g++ -std=c++17 -O2 -Wall -Wextra -Iideas/proto ideas/proto/sim/prepo.cpp ideas/proto/scheduler/queue.cpp ideas/proto/scheduler/cache.cpp ideas/proto/scheduler/preposition.cpp ideas/proto/sim/workload_mt.cpp ideas/proto/core/derive.cpp ideas/proto/core/optime.cpp ideas/proto/core/global_scheduler.cpp ideas/proto/core/workload.cpp -o ideas/proto/build/prepo.exe
```
(둘 다 경고 0으로 빌드됨.)

실행 (공통 CLI): `prog [iters] [Z] [eligibility] [evict_age] [global_age_cap] [seed_backlog]`
모든 측정: `iters=8000 Z=64`. A는 eligibility를 무시(캐시 항상 OFF)하되 동일 seed_backlog를 위해 같은 인자를 받음.

---

## 1. Head-to-head 대조 (matched backlog, 재튜닝 cap `8000 64 16000 200 1000 300`)

> 주: B는 **재튜닝된 global_age_cap=1000** 기준. (예전 cap=100 의 Σdev 17.018% / hbmHit 91.23% / p99 0.354M 은 **재튜닝 전 값** — cap=100 은 강제 교체 과다로 Σdev 가 부풀려졌다. 재튜닝 근거·스윕은 REPORT2.md.) A 는 글로벌 age-cap 의 영향을 받지 않으므로 수치 불변.

| 지표 | A (baseline / 노드-로컬 steering) | B (prepo / 글로벌 사전배치) |
|---|---|---|
| Σdev avg (idle) | **1.643%** | 8.323% |
| Σdev worst | **19.74%** | 101.33% |
| missAvgDev | 18.21% | 22.84% |
| hit% (배치 구성) | 99.22% | 70.77% |
| p99 tail (us) | 12,670,994.6 | **2,166,651.4** |
| hbmHit% (캐시) | 0.00% (캐시 OFF) | **81.83%** |
| ssdReload | 9,856 | 2,654 |
| recompute | 548 | **0** |
| savedR (재계산 절감) | 16,028,469,134 | **27,338,022,193** |
| max_wait | **8000 (= iters → starvation)** | 998 |
| forced | 0 (글로벌 age-cap 없음) | 1,102 (유한) |
| queue size | 8,806 | 258 |
| poolMean | 101,829 | 108,468 |
| resid (c/d/idle %) | 20.6 / 71.3 / 8.1 | 15.0 / 76.2 / 8.7 |
| returns | 10,404 | 14,607 |

핵심: **p99 tail ≈ 12.67M → 2.17M us (약 5.8× 단축)**. A는 글로벌 age-cap이 없어 `max_wait=8000`(= iters)으로 **기아(starvation) 발생** — 큐가 무한 적체(size 8,806). B는 `max_wait=998`, `forced=1,102`로 **대기 상한이 유한**. 단 B의 평균 idle(Σdev 8.32%)은 A보다 나쁘다. (cap 을 100→1000 으로 올려 강제 교체를 17,897→1,102 로 줄이자 Σdev 가 17.0%→8.3% 로 내려왔다 — 강제가 Σdev 의 주된 구동원이었음.)

---

## 2. evict_age 스윕 (캐시 히트 무릎) — `8000 64 16000 {evict} 100 300`

| evict_age | hbmHit% | ssdReload | p99 (us) | savedR |
|---|---|---|---|---|
| 25 | 5.31% | 14,097 | 369,116.5 | 30,612,017,535 |
| 50 | 12.94% | 12,960 | 369,116.5 | 30,616,695,027 |
| 100 | 18.37% | 12,152 | 369,116.5 | 30,622,879,520 |
| **200** | **91.23%** | **1,305** | **353,928.8** | 30,737,956,275 |
| 400 | 91.23% | 1,305 | 353,928.8 | 30,737,956,275 |

**무릎 = evict_age 200.** hbmHit이 100→200에서 18.37%→91.23%로 급등한다. 대기(`max_wait≈102`, queue 회전)가 evict_age를 넘으면 복귀 전 캐시가 축출되어 히트가 붕괴. evict_age가 대기 시간을 충분히 덮으면(≥200) 히트가 포화되고 200→400에서는 변화 없음(이미 모두 잔류).

---

## 3. global_age_cap 스윕 (공정성 ↔ 구성 분산) — `8000 64 16000 200 {cap} 300`

| global_age_cap | forced | max_wait | Σdev | hbmHit% | p99 (us) |
|---|---|---|---|---|---|
| 20 | 26,223 | 70 | 20.063% | 91.05% | 354,293.3 |
| 50 | 26,170 | 74 | 20.179% | 91.13% | 354,277.5 |
| **100** | **17,897** | **102** | **17.018%** | **91.23%** | **353,928.8** |
| 200 | 8,228 | 202 | 13.851% | 57.12% | 551,932.6 |
| 400 | 3,682 | 398 | 12.829% | 71.70% | 958,093.8 |

**무릎 = global_age_cap 100.** cap을 낮추면(20/50) forced 강제 교체가 늘고 max_wait는 짧아지지만 강제 set-change가 잦아 Σdev가 악화(20%대). cap을 올리면(200/400) Σdev는 다소 개선되나 **캐시 히트가 무너지고(91%→57%) p99가 급등**(354k→958k), max_wait도 cap을 따라 선형 증가. cap=100이 캐시 91% 유지 + p99 최저 + max_wait 유한의 균형점.

---

## 4. eligibility 스윕 (무엇을 캐시할지) — `8000 64 {elig} 200 100 300`

| eligibility | hbmHit% | ssdReload | Σdev | p99 (us) |
|---|---|---|---|---|
| 0 | **100.00%** | 0 | 17.018% | 353,928.8 |
| 16000 | 91.23% | 1,305 | 17.018% | 353,928.8 |
| 64000 | 53.48% | 6,925 | 17.018% | 355,592.9 |
| 256000 | 13.25% | 12,915 | 17.018% | 358,905.0 |

**무릎: eligibility를 높일수록 캐시 대상이 줄어 hbmHit 단조 감소(100→13%).** elig=0이면 전부 캐시되어 히트 100%·reload 0. elig 임계가 ctx(≈100K) 근처를 넘기 시작하는 64000부터 큰 폭으로 떨어진다. **Σdev는 eligibility에 불변**(17.018% 고정) — eligibility는 "무엇을 캐시하는가"만 바꿀 뿐 배치 구성(idle)에는 영향 없음을 확인.

---

## 5. seed_backlog 스윕 (발견된 Σdev ↔ 캐시 ↔ 대기 긴장) — `8000 64 16000 200 100 {backlog}`

| seed_backlog | Σdev | hbmHit% | max_wait | poolMean | p99 (us) | forced |
|---|---|---|---|---|---|---|
| 100 | 15.484% | 91.48% | 98 | 115,644 | 321,662.7 | 3,104 |
| 300 | 17.018% | 91.23% | 102 | 115,814 | 353,928.8 | 17,897 |
| **1000** | 18.631% | **0.21%** | 263 | 114,578 | 987,561.3 | 26,463 |
| 3000 | 17.777% | 0.00% | 796 | 112,755 | 2,630,727.4 | 26,386 |
| 8000 | 15.107% | 0.00% | 2,150 | 106,880 | 5,697,067.9 | 26,385 |

**캐시 절벽(cliff) = backlog 300 → 1000.** backlog≤300에서는 대기(`max_wait≈100`) < evict_age(200) → hbmHit≈91%. backlog 1000부터 대기가 evict_age를 추월(max_wait 263)하여 **hbmHit이 0.21%로 붕괴**, 이후 max_wait·p99가 backlog에 비례해 폭증(8000: max_wait 2,150, p99 5.7M). **Σdev는 backlog와 무관**(15~19% 범위에서 비단조) — Σdev가 평균 drift가 아니라 **고정 set의 per-batch 구성 분산**임을 확인(backlog를 키워도 개선되지 않음).

---

## 분석 (정직하게)

### B의 이득
- **p99 tail ~5.8× 단축**: 12.67M → 2.17M us (head-to-head, 재튜닝 cap=1000). A는 글로벌 age-cap 부재로 `max_wait=8000`(=iters) 기아 — B는 `max_wait=998`로 상한 유한.
- **캐시 히트 = 멀티턴 재계산 절약**: hbmHit 81.83%, recompute **0**(A는 548), savedR 27.3G(A 16.0G의 ~1.7배). 멀티턴 복귀 시 KV 재계산을 캐시로 흡수.
- **공정성**: forced·max_wait 모두 유한(1,102 / 998) vs A의 starvation(`max_wait=iters=8000`, queue size 8,806 무한 적체).
- **100K 평균 유지**: poolMean 108,468(ctx_balance≈100,169 대비 동급 규모 유지).

### B의 비용
- **Σdev(평균 idle)가 A보다 높다**: 1.643% → 8.323% (worst 19.74%→101.33%, 재튜닝 cap=1000 기준; 예전 cap=100 의 17.018% 는 강제 과다로 부풀려진 재튜닝 전 값). 노드가 순수 executor가 되어 **노드 re-selection을 제거**했기 때문 — 고정 set의 per-batch 분산을 매 라운드 능동 보정하지 못한다.
- **backlog 무관 확인**: seed_backlog 100~8000 전 구간에서 Σdev가 15~19%로 머물고 개선되지 않음 → 평균 drift가 아니라 **고정 구성 분산**. idea 문서의 "이득의 본질은 평균 idle이 아니라 tail/cache"와 정합하되, **Σdev(=idle)는 오히려 악화됨을 명시**.

### 캐시 cliff
대기 시간(≈ backlog / 회전율)이 evict_age를 넘으면 복귀 전 캐시가 축출되어 hbmHit이 붕괴 → evict_age 무릎(200)과 backlog 절벽(300→1000)이 동일 메커니즘. 두 스윕에서 교차 확인됨.

### 회전율 주의
디코더 장수명(max_tokens 256–4096 라운드)으로 글로벌 회전율이 낮다(returns 14,887 / iters 8000 ≈ 1.86/iter, 라운드 기준 ~3.6/round 수준). 따라서 backlog가 크면 대기↑ → 캐시↓·강제↑. **silent cap은 없으며**(forced로 명시적으로 노출됨) 이 한계를 그대로 드러낸다 — backlog≥1000에서 hbmHit≈0은 측정 그대로.

---

## 검증 질문 답 (idea 문서)

**① 복귀 jitter 흡수 최소 슬랙은?**
디코드는 결정론적(슬랙 0) — 측정상 miss는 전부 decode 측(miss b1 c/d=0/263181, b2 c/d=0/252928)이고 prefill(c) miss=0. stochastic한 부분은 **복귀-도착 타이밍**뿐이며, 이를 **글로벌 age-cap이 대기 상한으로 흡수**한다(cap=100 → max_wait=102). 별도 슬랙 버퍼 불필요.

**② 글로벌 제어 vs forward-pass 박자**
라운드 = balance_time_us × layers ≈ 2ms 규모. 회전율이 낮아 set-change가 드물어(forced 17,897 / 960,000 batches ≈ 1.9%) **제어 여유가 크다**. global_age_cap 스윕에서 cap을 올려도(set-change↓) 캐시·p99만 바뀔 뿐 제어가 박자를 못 따라가는 징후 없음.

**③ 캐시가 tail을 개선하는가?**
그렇다. head-to-head에서 캐시 ON(B)의 p99=353,928.8 us vs 캐시 OFF(A)의 p99=12,670,994.6 us. evict_age 스윕에서도 캐시가 살아있는 구간(≥200, hbmHit 91%)의 p99(353,928.8)가 캐시가 죽은 구간(evict 25~100, hbmHit 5~18%)의 p99(369,116.5)보다 낮다 — 캐시 잔류가 tail을 직접 단축.

---

## 경계 (가정 라벨)

- **100K (ctx_balance=100,169), 1.33TB, decode_count_target=62** 등은 **Llama70B + B200 derive 산출값**(측정 기준 operating point). 임의 워크로드 가정 아님.
- **gone_age=3000, offload_bw, think_gap** 은 **고정 가정 라벨**(이번 스윕 대상 아님). prepo CONFIG에 gone_age=3000으로 노출되며 본 보고의 5개 스윕(evict_age / global_age_cap / eligibility / seed_backlog + head-to-head)에는 포함하지 않았다.
- 모든 수치는 `iters=8000 Z=64`, `batches=960000` 단일 시드 실행의 측정값. 추정·외삽 없음.
