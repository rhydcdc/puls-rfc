# 스케줄러 3종 대조 측정 보고 (A=baseline vs B=prepo vs C=csched) — Plan 2

**1줄 요약:** global_age_cap 을 100→1000 으로 재튜닝하면 B 의 Sdev 가 17.0%→8.3% 로 내려와(강제 교체가 Sdev 의 주 구동원) A 와의 idle 격차가 좁혀지지만, 실 KPI 의 승자는 B — SLO goodput·TTFTmet 에서 A·C 를 크게 앞선다. C(=A 의 배치 방식 + B 의 age-cap·캐시)는 같은 cap·캐시에도 hbmHit~3%·TTFTmet~35% 로 B(82%·97%)에 크게 못 미친다: hole-healing 배치는 복귀/off-분포를 큐에 적체시켜(queue~8,800) 캐시가 살아나기 전에 축출당한다. 이는 HW 가 아니라 배치 방식의 풀-배수(pool-draining) 차이다.

## 빌드 / 실행

빌드 (repo 루트에서, 경고 0):

    baseline: g++ -std=c++17 -O2 -Wall -Wextra -Iideas/proto sim/baseline.cpp + queue/cache/workload_mt + core(derive,optime,steering,global_scheduler,workload) -> build/baseline.exe
    prepo:    sim/prepo.cpp + queue/cache/preposition/workload_mt + core(derive,optime,global_scheduler,workload) -> build/prepo.exe
    csched:   sim/csched.cpp + queue/cache/workload_mt + core(derive,optime,steering,global_scheduler,workload) -> build/csched.exe

실행 (공통 CLI): prog [iters] [Z] [eligibility] [evict_age] [global_age_cap] [seed_backlog]
모든 측정: iters=8000 Z=64, batches=960,000, 단일 시드. A 는 글로벌 age-cap 을 무시(CONFIG 라인에 라벨만 출력) — QUEUE-A 가 max_wait=8000, forced=0 으로 확인됨.

세 드라이버 정의:
- A (baseline) = 노드-로컬 steering 재선택(surplus 134 + ideal=hole healing), 글로벌 age-cap 없음, HBM 캐시 없음.
- B (prepo) = 글로벌 결정론 사전배치(124, surplus 없음), 글로벌 age-cap(pull_slot), HBM 캐시(124 기준 용량).
- C (csched) = A 의 배치 방식(steering 재선택 + surplus + healing) + B 의 글로벌 age-cap + B 의 캐시(134 기준 용량). 배치 방식 단독 효과 분리용.

## 1. global_age_cap 재튜닝 스윕 — 8000 64 16000 200 {cap} 300

### B (prepo)

| global_age_cap | Sdev | forced | max_wait | hbmHit% | TBT p99 (us) | TTFTmet% | SLO goodput (tok/s) |
|---|---|---|---|---|---|---|---|
| 100 | 17.018% | 17,897 | 102 | 91.23% | 3,588.5 | 96.9% | 3,079,974 |
| 250 | 13.489% | 6,433 | 251 | 62.56% | 3,951.6 | 96.9% | 3,236,947 |
| 500 | 10.573% | 2,721 | 501 | 74.37% | 3,417.6 | 96.6% | 3,380,231 |
| **1000** | **8.323%** | **1,102** | **998** | **81.83%** | **3,182.7** | **97.1%** | **3,534,768** |
| 2000 | 7.603% | 483 | 2,001 | 85.55% | 2,947.7 | 97.2% | 3,594,552 |
| 4000 | 6.344% | 195 | 4,001 | 87.08% | 3,011.8 | 97.1% | 3,598,078 |

### C (csched)

| global_age_cap | Sdev | forced | max_wait | hbmHit% | TBT p99 (us) | TTFTmet% | SLO goodput (tok/s) | queue |
|---|---|---|---|---|---|---|---|---|
| 100 | 2.529% | 26,134 | 2,415 | 2.70% | 2,157.4 | 35.3% | 2,049,402 | 8,827 |
| 250 | 2.529% | 26,134 | 2,415 | 2.70% | 2,157.4 | 35.3% | 2,049,402 | 8,827 |
| 500 | 2.432% | 25,699 | 2,420 | 2.68% | 2,157.4 | 35.7% | 2,083,744 | 8,828 |
| **1000** | 2.234% | 24,556 | 2,420 | 3.19% | 2,157.4 | 34.6% | 2,057,535 | 8,829 |
| 2000 | 2.011% | 21,885 | 2,420 | 3.70% | 2,157.4 | 34.6% | 2,066,830 | 8,820 |
| 4000 | 1.731% | 6,609 | 4,001 | 12.36% | 2,136.0 | 50.8% | 2,495,999 | 8,813 |

### 채택값 + 근거

**채택 global_age_cap = 1000** (B 기준 무릎).

B 에서 cap 을 올릴수록 Sdev down, forced down 가 단조로 나타나고(17.0->6.3% / 17,897->195) max_wait 는 cap 에 선형으로 늘어난다(102->4,001). 즉 기존 cap=100 은 강제 교체 과다(forced 17,897)로 Sdev 를 17% 까지 부풀렸다 — cap up 하면 Sdev down, forced down 단 max_wait up 의 knee 구조다.

cap=1000 근거(수치):
- Sdev 이득의 대부분 확보: 17.0->8.3% (전체 17.0->6.3% 낙폭의 약 79%) 가 cap=1000 에서 달성. 1000->4000 추가 이득은 8.3->6.3% (2%p) 뿐.
- forced 1,102 (fills 대비 ~0.1%) 로 작음 — set-change 드물어 제어 여유.
- max_wait 998 로 유한·유계: cap=2000/4000 은 Sdev·hbmHit·goodput 이 미세하게만 더 좋아지는 대가로 max_wait 를 2,001·4,001 로 2~4x 키운다(tail 대기 악화). cap=1000 이 이득 대부분 + 대기 절반 의 균형.
- hbmHit 81.83%·SLO goodput 3.53M 로 캐시·goodput 이 포화 근처(2000: 85.6%/3.59M).

C 는 cap 을 올려도 queue 가 ~8,800 으로 고착되어(아래 분석) Sdev 만 미세 개선될 뿐 hbmHit·goodput 이 거의 안 움직인다(cap=4000 에서야 hbmHit 12%/goodput 2.50M 로 약간 반응). C 는 cap 으로 구제되지 않음 — 같은 cap=1000 에서 대조한다.

주의(측정 정직성): 사전 컨텍스트의 cap=2000 에서 B Sdev~1.4%, forced~77 은 본 빌드/시드에서 재현되지 않았다. 측정값은 cap=2000 에서 Sdev=7.603%, forced=483. 방향(cap up -> Sdev down, forced down, max_wait up)은 일치하나 절대 수치는 다르다. 본 보고는 측정값만 기재한다.

## 2. Head-to-head A vs B vs C — 채택 cap=1000 (8000 64 16000 200 1000 300)

| 지표 | A (baseline) | B (prepo) | C (csched) |
|---|---|---|---|
| TBT mean (us) | 2,044.8 | 2,201.3 | 2,048.3 |
| TBT p99 (us) | **2,136.0** | 3,182.7 | 2,157.4 |
| TTFT mean (us) | 4,202,430.1 | **864,842.5** | 5,771,791.3 |
| TTFT p99 (us) | 16,390,662.6 | **10,497,548.6** | 14,703,807.3 |
| SLO goodput (tok/s) | 2,908,175 | **3,534,768** | 2,057,535 |
| TTFTmet% | 67.7% | **97.1%** | 34.6% |
| PIMwin2 | **33.5%** | 0.4% | 31.0% |
| win3 (pim/gpua/ffn %) | 34/66/0 | 0/100/0 | 31/63/6 |
| goodput TBT (tok/s) | 3,912,216 | 3,659,813 | 3,896,360 |
| — 보조 — | | | |
| Sdev avg | **1.643%** | 8.323% | 2.234% |
| Sdev worst | 19.74% | 101.33% | 55.16% |
| hbmHit% | 0.00% (OFF) | **81.83%** | 3.19% |
| forced | 0 | 1,102 | 24,556 |
| max_wait | 8,000 (starvation) | **998** | 2,420 |
| queue size | 8,806 | **258** | 8,829 |
| poolMean | 101,829 | 108,468 | 114,092 |
| resid (c/d/idle %) | 20.6/71.3/8.1 | 15.0/76.2/8.7 | 20.3/69.2/10.5 |
| returns | 10,404 | 14,607 | 9,685 |
| recompute | 548 | **0** | 96 |
| savedR | 16.03G | **27.34G** | 17.23G |

공통 라벨: T_ttft=5,000,000 us, T_tbt~2,637.1 us (~1.3 x round_us, round_us~2,029).

## 3. SLO 임계 라벨 (재컴파일 없음)

slo_tbt_mult 는 컴파일 타임 상수라 본 측정에서 스윕하지 않았다. 위 표의 SLO goodput·TTFTmet% 는 기본 SLO 기준이며, 임계 라벨은:
- T_ttft = 5,000,000 us (TTFT 충족 판정 임계, 가정 라벨)
- T_tbt ~ 2,637.1 us ~ 1.3 x round_us (round_us ~ 2,029 us, decode 라운드 박자에서 파생)

이 임계들은 가정 라벨이며 HW·SLA 정책에 종속된다. goodput 비교의 상대 순위는 임계 절대값이 아니라 분포(TBT·TTFT)에서 나온다.

## 4. idle-win 표 (채택 cap=1000)

| 드라이버 | PIMwin2 (2-way) | win3 pim | win3 gpua | win3 ffn |
|---|---|---|---|---|
| A | 33.5% | 34% | 66% | 0% |
| B | 0.4% | 0% | 100% | 0% |
| C | 31.0% | 31% | 63% | 6% |

PIMwin2 = PIM 이 이기면 PIM 이 숨는다(t_pim < t_gpu_a -> PIM hidden). A·C ~31~34%, B ~0.4% 로 B 만 PIM 노출.

## 5. 분석 (정직하게)

### 가설 1 — 재튜닝 후 Sdev 수렴, 강제가 주범

cap=1000 에서 A 1.643% · C 2.234% · B 8.323%. 강제 교체를 줄이면(B forced 17,897->1,102) Sdev 가 17.0->8.3% 로 크게 내려와 강제가 B Sdev 의 주된 구동원이었음이 확인된다(B 스윕에서 forced 와 Sdev 가 동반 단조 하강).

다만 완전 수렴은 아니다: 강제가 거의 0 인 영역(cap=4000, forced 195)에서도 B Sdev=6.344% 로 A(1.643%)·C(1.731%, cap=4000) 보다 높다. 즉 강제를 빼고 남은 잔여 Sdev 에서는 steering(A/C)이 명확히 우위다 — A·C 는 매 라운드 노드 re-selection 으로 고정 set 의 per-batch 분산을 능동 보정하지만, B 는 사전배치 후 노드가 순수 executor 라 잔여 구성 분산을 흡수 못 한다. C 의 Sdev(2.234%)가 B(8.323%)의 약 1/4 인 것이 같은 cap·캐시 조건에서 배치 방식만 바꾼 직접 증거다.

### 가설 2 — PIMwin2 / TBT 는 Sdev·poolMean 의 systematic 편향과 연동

B 는 PIMwin2=0.4%, TBT p99=3,182.7 (A·C 의 ~2,150 대비 높음). B 의 배치는 사전배치 풀평균(poolMean 108,468) 위에서 systematic 하게 target 보다 약간 높게 앉아 t_pim 이 t_gpu_a 보다 일관되게 근소 초과 -> PIM 이 거의 항상 노출(win3 gpua 100%). 반면 A·C 는 hole-healing 으로 target 근처를 맞춰(steering) t_pim 이 이기는 라운드가 ~1/3 발생 -> PIMwin2 31~34%. B 의 낮은 PIMwin2 와 높은 TBT p99 는 같은 systematic 편향의 양면이다. (단 이 TBT 열위는 절대적으로 작다 — B p99 3.18ms 도 T_tbt 2.64ms 근방이라 TBT 자체로 SLO 를 깨는 수준은 아니며, 실 goodput 은 TTFT 가 좌우한다.)

### 배치 방식의 풀-배수(pool-draining) 효과 — C 의 핵심 열위

같은 글로벌 age-cap·같은 캐시 메커니즘인데도 C 의 hbmHit~3% vs B~82%, TTFTmet 35% vs 97%, SLO goodput 2.06M vs 3.53M. 원인은 배치 방식이 풀을 비우는 방식이 다르기 때문:

- B (centering-preposition) 는 풀을 평균 중심으로 고르게 배수 -> 복귀(returns 14,607)·off-분포 요청을 evict_age(200) 내에 서빙 -> 캐시 잔류분을 히트(hbmHit 82%). queue 가 258 로 작게 유지.
- C (hole-healing) 는 ideal=hole 만 끌어와 -> 복귀(누적/장수명)·off-분포 요청이 큐에 적체. queue~8,829, max_wait 2,420 >> evict_age 200 -> 복귀 요청이 age-cap(1000)이 강제로 밀어넣기 전에 이미 캐시에서 축출 -> hbmHit 3%. 강제 교체만 폭증(forced 24,556).

C 의 hbmHit vs global_age_cap (배치 방식 효과 입증):

| cap | C hbmHit% | C forced | C Sdev | C SLO goodput |
|---|---|---|---|---|
| 100 | 2.70% | 26,134 | 2.529% | 2,049,402 |
| 500 | 2.68% | 25,699 | 2.432% | 2,083,744 |
| 1000 | 3.19% | 24,556 | 2.234% | 2,057,535 |
| 2000 | 3.70% | 21,885 | 2.011% | 2,066,830 |
| 4000 | 12.36% | 6,609 | 1.731% | 2,495,999 |

C 는 cap 전 구간에서 hbmHit 이 ~3% 에 갇혀 있고 cap=4000 에서야 12% 로 약간 반응(forced 26k->6.6k 로 떨어지며 큐 적체 일부 완화). 즉 C 의 캐시 열위는 age-cap 튜닝으로 구제되지 않는다 — hole-healing 배치 자체가 복귀/off-분포를 적체시키는 풀 동역학이다. (B 는 같은 cap=4000 에서 hbmHit 87%.) 이는 HW 파생값이 아니라 순수 배치-방식 효과다.

측정 정직성: 사전 컨텍스트는 lower cap -> C 가 복귀를 더 일찍 서빙 -> hbmHit up 을 예상했으나, 측정은 반대 — C hbmHit 은 cap 과 함께 증가(2.70%->12.36%). cap 을 낮추면 강제 교체가 더 잦아질 뿐 큐 적체(~8,800)는 해소되지 않아 복귀 서빙이 빨라지지 않는다. 본 보고는 측정 방향을 따른다.

### 종합 — 실 KPI 의 승자

- SLO goodput·TTFTmet 의 명확한 승자는 B (3.53M tok/s, 97.1%) — A(2.91M, 67.7%)·C(2.06M, 34.6%) 를 크게 앞선다. B 의 작은 큐(258)·짧은 max_wait(998)·높은 hbmHit(82%) 가 TTFT 분포를 압도적으로 단축(TTFT mean 0.85M vs A 4.2M / C 5.77M).
- idle(Sdev)·TBT p99·PIMwin2 에서는 A 가 최상 (Sdev 1.64%, TBT p99 2.14ms, PIMwin2 33.5%) 이지만, A 는 글로벌 age-cap 부재로 max_wait=8000(=iters) 기아 -> 큐 무한 적체(8,806) -> TTFT·goodput 손해.
- C 는 최악의 조합 — A 의 좋은 idle/TBT 는 일부 물려받되(Sdev 2.23%, TBT p99 2.16ms, PIMwin2 31%), 배치 방식 탓에 캐시가 죽고(hbmHit 3%) 큐가 폭주(8,829)하여 goodput·TTFTmet 이 셋 중 꼴찌. age-cap·캐시를 B 에서 빌려와도 배치 방식(hole-healing)이 그 이득을 무효화한다.

결론: idle/Sdev 우위는 batch 방식(steering re-select, A·C)에서 오지만, 실 서빙 KPI(SLO goodput·TTFT)는 풀을 고르게 배수하는 centering-preposition(B)가 지배한다. idle 최소화와 tail/cache/공정성 확보는 별개이며, 본 워크로드에서 후자가 goodput 을 결정한다.

## 6. 경계 (가정 라벨)

- ctx_balance=100,169, decode_count_target=62, decode_pool=134(A/C)·124(B), round_us~2,029 는 Llama70B + B200 derive 산출값(deployed operating point). 임의 워크로드 가정 아님.
- T_ttft=5,000,000 us, T_tbt~2,637.1 us(~1.3 x round_us), gone_age=3000, eligibility=16000, evict_age=200 은 고정 가정 라벨 — 본 보고 스윕(global_age_cap)에는 포함되나 SLO 임계는 재컴파일이 필요해 스윕하지 않았다.
- 모든 수치는 iters=8000 Z=64, batches=960,000 단일 시드 실행의 측정값. 추정·외삽 없음. 사전 컨텍스트와 어긋난 두 지점(B cap=2000 Sdev, C hbmHit vs cap 방향)은 측정값을 따랐고 본문에 명시했다.
