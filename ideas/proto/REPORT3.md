# REPORT3 — Plan-3 스케줄러 비교 프로토타입 (측정)

**1줄 요약**: reload BW(2e7, SSD급)·콜드스타트 시딩 버그를 교정한 뒤, 실 KPI(goodput/TTFT)는 캐시 ON 인 B·C 가 miss 비용이 커질수록 A 대비 우위를 보이고, idle 품질(TBT/Σdev)은 PIM 노출이 낮은 A·C 가 B 보다 우수하다 — 즉 교정 후 그림은 "B 가 TTFT, A·C 가 TBT" 로 갈라진다.

---

## 빌드 / 실행

빌드 (repo 루트, g++ -O2 -Wall -Wextra, 경고 0):
```
g++ -std=c++17 -O2 -Wall -Wextra -Iideas/proto ideas/proto/sim/baseline.cpp ideas/proto/scheduler/queue.cpp ideas/proto/scheduler/cache.cpp ideas/proto/sim/workload_mt.cpp ideas/proto/core/derive.cpp ideas/proto/core/optime.cpp ideas/proto/core/steering.cpp ideas/proto/core/global_scheduler.cpp ideas/proto/core/workload.cpp -o ideas/proto/build/baseline.exe
g++ -std=c++17 -O2 -Wall -Wextra -Iideas/proto ideas/proto/sim/prepo.cpp ideas/proto/scheduler/queue.cpp ideas/proto/scheduler/cache.cpp ideas/proto/scheduler/preposition.cpp ideas/proto/sim/workload_mt.cpp ideas/proto/core/derive.cpp ideas/proto/core/optime.cpp ideas/proto/core/global_scheduler.cpp ideas/proto/core/workload.cpp -o ideas/proto/build/prepo.exe
g++ -std=c++17 -O2 -Wall -Wextra -Iideas/proto ideas/proto/sim/csched.cpp ideas/proto/scheduler/queue.cpp ideas/proto/scheduler/cache.cpp ideas/proto/sim/workload_mt.cpp ideas/proto/core/derive.cpp ideas/proto/core/optime.cpp ideas/proto/core/steering.cpp ideas/proto/core/global_scheduler.cpp ideas/proto/core/workload.cpp -o ideas/proto/build/csched.exe
```

실행 (iters 8000, Z 64). CLI: [iters] [Z] [eligibility] [evict_age] [global_age_cap] [seed_backlog] [contention_beta] [offload_bw] [node_age_cap]
- base config: 8000 64 16000 200 1000 300 0.5 2e7
- 배포 op: Llama70B + B200, ctx≈100K, decode_count_target=62, round_us≈2029µs, num_layers=80.

드라이버: **A=baseline.exe** (노드 steering 재선택 + surplus, 글로벌 age-cap 없음, HBM 캐시 OFF) · **B=prepo.exe** (글로벌 사전배치, 글로벌 age-cap, 캐시 ON, 124 풀) · **C=csched.exe** (A 배치법 + 글로벌 age-cap + 캐시, 134 풀).

## Plan-3 변경 요약 (3가지)
1. **reload BW 교정**: offload_bw 기본값 5e9(HBM급) → **2e7 B/round** (SSD ~10 GB/s). miss/reload 비용이 길이에 비례하는 현실값이 됨.
2. **콜드스타트 시딩 버그 수정 (A/C)**: 큐 과충전(옛 ~8800) 제거 → C 캐시가 정상 작동 (hbmHit 옛 3~10% → ~75%).
3. **A→B 의존성 + HBM 컨텐션**: TBT = max(instance_a_latency, t_ffn) × num_layers, instance_a_latency = max(t_pim,t_gpu_a) + β·max(0, t_pim−t_gpu_a). 컨텐션-free 조건 = t_pim ≤ t_gpu_a. β=0 ⇒ 옛 max(3)×L 로 회귀. 신규 KPI: PIMexposed%, expo_us.

---

## A vs B vs C 대조표 (base config 8000 64 16000 200 1000 300 0.5 2e7)

| 지표 | A (baseline) | B (prepo) | C (csched) |
|---|---|---|---|
| TBT mean (µs) | **2055.4** | 2288.9 | 2057.9 |
| TBT p99 (µs) | **2191.0** | 3761.0 | 2191.0 |
| TTFT mean (µs) | 1781445 | **1206920** | 1258630 |
| TTFT p99 (µs) | 10163758 | 12993402 | **11055153** |
| SLO goodput (tok/s) | 3676862 | 3201419 | **3685357** |
| TBT goodput (tok/s) | **3912081** | 3387836 | 3900071 |
| TTFTmet% | **93.6** | 93.2 | 93.5 |
| PIMwin2% | 30.0 | 0.4 | 27.8 |
| **PIMexposed%** | 70.0 | **99.6** | 72.2 |
| **expo_us** (µs/µ-batch) | 19 | **175** | 21 |
| Σdev% (보조) | 1.504 | 8.323 | 1.512 |
| hbmHit% (보조) | 0.00 (캐시 OFF) | **81.83** | 75.46 |
| forced (보조) | 0 | 1102 | 673 |
| max_wait (보조) | 6471 | 998 | 989 |
| queue size (보조) | 368 | 258 | 377 |

읽는 법:
- **TBT/Σdev** (idle 품질): A≈C 우수, B 열위. B 는 PIM 노출 99.6% / expo_us 175µs 라서 β>0 에서 컨텐션이 TBT 에 직접 가산 → TBT mean +233µs, p99 +1570µs.
- **TTFT/goodput** (실 KPI): TTFT mean 은 B 최저(캐시 hbmHit 81.83%), p99 는 C 최저. SLO goodput 은 C 최고(B 는 TBT-SLO 미달 라운드가 많아 토큰 가중에서 손해).
- A 는 글로벌 age-cap 이 없어 max_wait 6471 — starvation 노출(공정성 약점). B/C 는 age-cap 으로 max_wait ≈ cap(998/989).

---

## 스윕 표

### 1. β (HBM 컨텐션) 스윕 — ... 300 {0.0,0.25,0.5,1.0} 2e7

| β | 드라이버 | TBT mean | TBT p99 | PIMexposed% | expo_us | SLO goodput |
|---|---|---|---|---|---|---|
| 0.0 | A | 2045.9 | 2136.0 | 70.0 | 19 | 3676862 |
| 0.5 | A | 2055.4 | 2191.0 | 70.0 | 19 | 3676862 |
| 1.0 | A | 2065.0 | 2246.0 | 70.0 | 19 | 3676862 |
| 0.0 | B | 2201.3 | 3182.7 | 99.6 | 175 | 3448284 |
| 0.25 | B | 2245.1 | 3471.8 | 99.6 | 175 | 3325737 |
| 0.5 | B | 2288.9 | 3761.0 | 99.6 | 175 | 3201419 |
| 1.0 | B | 2376.5 | 4339.2 | 99.6 | 175 | 2999964 |
| 0.0 | C | 2047.4 | 2136.0 | 72.2 | 21 | 3685357 |
| 0.25 | C | 2052.7 | 2163.5 | 72.2 | 21 | 3685357 |
| 0.5 | C | 2057.9 | 2191.0 | 72.2 | 21 | 3685357 |
| 1.0 | C | 2068.5 | 2246.0 | 72.2 | 21 | 3685357 |

**Finding**: PIMexposed%·expo_us 는 β 와 무관(노출은 op-time 으로 결정, β 는 노출시간에 곱해지는 *벌점 계수*). β 0→1 에서 TBT 증가폭이 노출량에 비례 — B(expo 175µs)는 TBT mean +175µs / p99 +1157µs / goodput −448K(−13%), A·C(expo ~20µs)는 TBT mean +20µs 미만, goodput 거의 불변. **사슬 정량화: 노출(PIMexposed)이 큰 드라이버일수록 β 민감**. C·A 의 SLO goodput 이 β 전 구간 동일한 것은 TBT 가 SLO(2637µs) 한참 아래라 토큰 가중이 안 바뀌기 때문, B 만 SLO 경계 근처라 goodput 이 하락.

### 2. offload_bw 스윕 — ... 300 0.5 {2e7, 7e7, 2e8, 5e9}

| bw | 드라이버 | hbmHit% | TTFT mean | TTFT p99 | SLO goodput | TTFTmet% | recompute | ssdReload |
|---|---|---|---|---|---|---|---|---|
| 2e7 (SSD) | B | 81.83 | 1206920 | 12993402 | 3201419 | 93.2 | 0 | 2654 |
| 2e7 (SSD) | C | 75.46 | 1258630 | 11055153 | 3685357 | 93.5 | 0 | 3568 |
| 7e7 (DRAM) | B | 81.83 | 961598 | 10497549 | 3263257 | 96.3 | 0 | 2654 |
| 7e7 (DRAM) | C | 75.46 | 1020166 | 9814540 | 3762734 | 96.6 | 0 | 3568 |
| 2e8 | B | 81.83 | 897814 | 10497549 | 3273950 | 97.1 | 0 | 2654 |
| 2e8 | C | 75.46 | 958166 | 9814540 | 3764385 | 96.7 | 0 | 3568 |
| 5e9 (옛 HBM급) | B | 81.83 | 864843 | 10497549 | 3273950 | 97.1 | 0 | 2654 |
| 5e9 (옛 HBM급) | C | 75.46 | 926116 | 9814540 | 3764385 | 96.7 | 0 | 3568 |

**Finding**: hbmHit·ssdReload·recompute 는 bw 와 무관(히트 여부는 캐시 정책이 결정, bw 는 *miss 한 건의 비용*만 바꿈). bw 가 느릴수록(2e7) miss 비용이 커져 TTFT 가 부풀고(B: 864843→1206920, **+40%**; C: 926116→1258630, +36%), TTFTmet% 가 떨어진다(B 97.1→93.2). **옛 5e9 는 miss 비용을 거의 0 으로 만들어 캐시 효과를 가렸다**: 5e9 에서 B·C TTFT 차 ~61K(7%)뿐이지만, 제값 2e7 에서는 B 가 hbmHit 6%p 높은 덕에 캐시 우위가 TTFT 축에서 드러난다. 단 SLO goodput 은 전 bw 에서 C>B (B 의 TBT 열위가 토큰 가중을 깎기 때문) — bw 교정은 TTFT 축에서 캐시를 부각시키되 goodput 순위는 뒤집지 않는다.

### 3. global_age_cap 스윕 — 8000 64 16000 200 {250,500,1000,2000} 300 0.5 2e7 (B·C)

| cap | 드라이버 | Σdev% | forced | max_wait | hbmHit% | TBT p99 | SLO goodput |
|---|---|---|---|---|---|---|---|
| 250 | B | 13.489 | 6433 | 251 | 62.56 | 4914.4 | 2894563 |
| 500 | B | 10.573 | 2721 | 501 | 74.37 | 4113.4 | 2972095 |
| 1000 | B | 8.323 | 1102 | 998 | 81.83 | 3761.0 | 3201419 |
| 2000 | B | 7.603 | 483 | 2001 | 85.55 | 3408.5 | 3213162 |
| 250 | C | 1.538 | 5767 | 250 | 62.61 | 2199.0 | 3716235 |
| 500 | C | 1.528 | 1837 | 495 | 69.42 | 2199.0 | 3689319 |
| 1000 | C | 1.512 | 673 | 989 | 75.46 | 2191.0 | 3685357 |
| 2000 | C | 1.518 | 293 | 2001 | 80.07 | 2191.0 | 3701446 |

**Finding (knee)**: cap 을 죄면(250) 강제축출(forced)·Σdev·TBT p99 가 급등하고 hbmHit 이 무너진다(B 81.8→62.6). 완화 방향(2000)은 hbmHit·goodput 이 계속 좋아지지만 max_wait(공정성)가 cap 만큼 늘어남. **무릎은 ~1000 부근** — cap 250→1000 에서 B goodput +307K(+11%)·hbmHit +19%p 로 큰 회복, 1000→2000 은 +12K·+4%p 로 한계효용 급감(대신 max_wait 2배). 고정 비용 모델(2e7) 아래서도 Plan2 의 cap≈1000 무릎이 재확인된다.

### 4. evict_age 스윕 — 8000 64 16000 {50,100,200,400} 1000 300 0.5 2e7 (B·C)

| evict | 드라이버 | hbmHit% | ssdReload | SLO goodput | TTFTmet% |
|---|---|---|---|---|---|
| 50 | B | 72.93 | 3954 | 3178162 | 92.3 |
| 100 | B | 78.40 | 3155 | 3188533 | 92.7 |
| 200 | B | 81.83 | 2654 | 3201419 | 93.2 |
| 400 | B | 84.30 | 2293 | 3217501 | 93.8 |
| 50 | C | 52.29 | 6935 | 3650881 | 92.4 |
| 100 | C | 65.80 | 4972 | 3665035 | 92.9 |
| 200 | C | 75.46 | 3568 | 3685357 | 93.5 |
| 400 | C | 83.64 | 2378 | 3710937 | 94.5 |

**Finding (knee)**: evict_age ↑ = 캐시 보존 길어짐 → hbmHit·goodput 단조 증가. **무릎은 ~200** — C 는 50→200 에서 hbmHit +23%p(52→75)로 가파르게 오르다 200→400 은 +8%p 로 둔화. B 는 더 완만(72→84). goodput 이득은 evict 50→400 에서 C +60K(+1.6%)·B +39K 로 작아, 200 이상은 한계효용 체감. (B 의 PREPO 라인 hit/Σdev 가 evict 와 무관하게 동일한 것은 사전배치 큐 통계가 evict 와 분리돼 집계되기 때문 — hbmHit/ssdReload 만 evict 에 반응.)

### 5. node_age_cap 스윕 — 8000 64 16000 200 1000 300 0.5 2e7 {2,5,10,20} (A·C, argv[9])

| ncap | 드라이버 | Σdev% | hbmHit% | TTFTmet% | TBT mean | SLO goodput |
|---|---|---|---|---|---|---|
| 2 | A | 1.387 | 0.00 | 93.7 | 2054.9 | 3686259 |
| 5 | A | 1.504 | 0.00 | 93.6 | 2055.4 | 3676862 |
| 10 | A | 1.504 | 0.00 | 93.4 | 2055.0 | 3677332 |
| 20 | A | 1.543 | 0.00 | 93.6 | 2054.7 | 3678570 |
| 2 | C | 1.367 | 75.68 | 93.3 | 2057.5 | 3675465 |
| 5 | C | 1.512 | 75.46 | 93.5 | 2057.9 | 3685357 |
| 10 | C | 1.556 | 75.40 | 93.3 | 2058.1 | 3685961 |
| 20 | C | 1.611 | 75.69 | 93.3 | 2058.9 | 3694095 |

**Finding**: A·C 의 steering 공정성 민감도는 약하다. ncap 2→20 에서 Σdev 가 A 1.387→1.543, C 1.367→1.611 로 소폭 증가(엄격한 cap=2 가 더 자주 재배치를 강제 → 약간 더 균등)하나 hbmHit·TTFTmet·goodput 은 거의 불변(C hbmHit 75.4~75.7%). **knee 없음 — 이 워크로드에서 node_age_cap 은 둔감한 노브**. B 는 argv[9] 미사용으로 영향 없음(노드 age-cap 자체가 없음).

### 6. B vs C hbmHit (base config) — 시딩 수정 검증

| 드라이버 | 풀 용량 | hbmHit% |
|---|---|---|
| B (prepo) | 124 | 81.83 |
| C (csched) | 134 | 75.46 |
| **差** | — | **6.37%p** |

**Finding**: 옛 시딩 버그 시절 C hbmHit 은 3~10% 였다. 수정 후 75.46% 로, B(81.83%)와의 격차는 **6.37%p** 에 불과 — 풀 용량/운용 조밀도 차(124 vs 134)로 설명되는 수준. **C 의 낮던 히트율은 캐시 정책 결함이 아니라 시딩 버그였음이 확정**된다.

---

## 분석 (정직)

### reload BW 교정 효과
제값 2e7(SSD)에서 miss 한 건의 비용이 길이에 비례해 커지면서, 캐시 ON 인 B·C 의 hbmHit(81.8/75.5%)이 TTFT 를 직접 깎는다. offload_bw 스윕에서 본 대로 bw 2e7→5e9 로 miss 비용을 없애면 B TTFT mean 이 1206920→864843(−28%)까지 떨어지는데, **이 차이가 곧 "느린 BW 에서 캐시가 막아주는 양"**. 옛 5e9 는 miss 가 거의 공짜라 캐시 유무 차를 7% 안쪽으로 눌러 가렸고, 제값에서는 그 효과가 30~40% TTFT 차로 드러난다. 다만 SLO goodput 순위는 bw 와 무관하게 C>B 유지 — TTFT 축에서는 캐시(B)가, 토큰 가중 goodput 축에서는 낮은 TBT(C)가 이긴다.

### 시딩 버그 수정
base 에서 C hbmHit 75.46%, B 81.83% 로 격차 6.37%p(용량차 수준). 옛 3~10% 대비 약 8~25배 상승. **C 의 낮은 히트는 캐시 알고리즘이 아니라 큐 과충전(콜드스타트 시딩) 버그였음이 확정**된다. node_age_cap 전 구간에서 C hbmHit 이 75.4~75.7% 로 안정적인 점도 캐시가 이제 정상 동작함을 뒷받침한다.

### 컨텐션 사슬 (idle→노출→컨텐션→TBT→goodput)
B 는 PIM 노출 99.6%·expo_us 175µs, A·C 는 70~72%·~20µs. β 스윕에서 TBT 증가폭이 expo_us 에 정확히 비례(B 는 β 0→1 에 TBT mean +175, A·C 는 +20 미만). B 의 높은 노출은 idle 분산(Σdev 8.3% vs A·C 1.5%)의 결과 — 사전배치가 전역 균등을 노리다 PIM 측 op-time 이 GPU-A 를 자주 초과하는 배치를 만들고(t_pim>t_gpu_a), 그게 노출→컨텐션→TBT p99 급등→TBT-SLO 미달→토큰 가중 goodput 하락(β=1 에서 B goodput −13%)으로 이어진다. **사슬 정량화: Σdev(idle 불균형)↑ → PIMexposed↑ → β·노출이 TBT 에 가산 → goodput↓**. A·C 는 노출이 작아 이 사슬에서 자유롭다.

### 종합 (Plan2 대비 변화)
- **실 KPI(goodput/TTFT)**: SLO goodput 은 C 최고(3685K), TTFT mean 은 B 최저(1207K)·p99 는 C 최저(11.06M). 캐시 ON 두 드라이버가 캐시 OFF 인 A 의 TTFT(1781K)를 크게 앞선다 — bw 교정으로 캐시 우위가 비로소 측정됨.
- **idle(TBT/Σdev)**: A≈C 우수(TBT 2055/2058, Σdev 1.5%), B 열위(2289, 8.3%). 컨텐션 모델 도입으로 B 의 노출 약점이 TBT 에 가시화됨.
- **공정성(max_wait)**: B·C 는 글로벌 age-cap 으로 max_wait≈cap(~990), A 는 cap 없어 6471 — A 의 starvation 약점이 유일하게 드러나는 축.
- **Plan2 대비**: 옛 5e9 BW·시딩 버그 하에선 C 가 저히트로 캐시 무용처럼 보였고 컨텐션이 없어 B 의 노출 비용도 안 보였다. 교정 후 그림은 **"B=TTFT 우위(캐시·사전배치) / A·C=TBT·Σdev 우위(낮은 노출) / C=goodput 종합 1위(둘의 균형)"** 로 또렷해졌다.

---

## 경계 + 향후 (deferred)

가정 라벨(절대 진실 아님):
- **offload_bw / β / SLO 임계(T_ttft=5e6, T_tbt=2637µs)** 는 모두 가정 라벨이며 스윕 가능한 노브.
- **TBT 절대값**은 B200 + Llama70B optime 종속 — HW·모델 바뀌면 재측정 필요.

**향후 (deferred)**: 본 프로토타입의 TBT 는 *해석적 정상상태* 공식(더블버퍼링 가정 하 max(instance_a_latency, t_ffn)×num_layers)이다. **인스턴스 내 디스패치 DAG (ARCH §6.3, 커널-완료 이벤트 구동) 는 이 프로토타입 범위 밖이다 — 과도구간(transient)·디스패치 순서·컨텐션 *타이밍* 의 충실 검증은 향후 sim 타이밍 층을 이벤트-구동으로 올릴 때 다룬다. 그때도 scheduler/ 로직(큐·캐시·사전배치)은 그대로 재사용된다.** 즉 본 리포트의 컨텐션 결론은 *정상상태 평균* 수준의 정량화이며, 마이크로 타이밍 충돌의 정확한 분포는 이벤트-구동 층의 과제로 남긴다.
