# Plan 2 — 배치 선택 방식 정밀도 격리 + 실 KPI(TBT·TTFT·SLO goodput) 측정

> Plan 1([predictive_prepositioning_scheduler.md](predictive_prepositioning_scheduler.md))의 두 결함을 교정한다:
> ① A·B 비교가 *불공정*했다(A 엔 글로벌 age-cap·캐시 없음 — 현 as-is). B 의 Σdev 가 높은 게
> *배치 선택 방식* 탓인지, *글로벌 age-cap 의 off-fit 강제* 탓인지, *캐시* 탓인지 섞여 있다.
> ② **idle 에 과집중**했다. idle/Σdev 는 proxy 일 뿐, 진짜 KPI 는 **TBT·TTFT·SLO goodput**.

---

## 1. 회고 — Plan 1 실측 (REPORT.md)
- B(글로벌 pre-positioning): **p99 tail ~36× 단축 + 캐시 hbmHit ~91%(재계산 0) + 공정성 bounded**.
- 비용: **Σdev(idle) 10~17% 로 A(1.6%)보다 악화** — backlog 무관(고정 set 분산).
- 즉 B 는 idle 을 내주고 tail/cache 를 얻었다. 그런데 **그 비교의 A 엔 글로벌 age-cap·캐시가 없었다.**

## 2. 두 가설

> **정정 (2026-06 Q&A 실측).** 초기 가설 "B 는 set 을 고정하고 steering 을 못 한다"는 **틀렸다.**
> B 도 글로벌 풀에서 `ideal = (100K타깃 − Σ남은) / 남은슬롯` 으로 **능동·고자유도 재선택**을 한다
> (선택지는 134 상주뿐인 A 보다 오히려 넓다). 코드 검증 + global_age_cap 스윕(동일 조건)으로 Σdev
> 원인을 분해:
>
> | global_age_cap | forced | Σdev |
> |---|---|---|
> | 100 (기존 기본) | 17,897 | 17.0% |
> | 500 | 2,721 | 10.6% |
> | 2000 | 483 | 7.6% |
> | ∞ (강제 0) | 0 | 6.3% |

1. **B 의 높은 Σdev = (주) 글로벌 age-cap 강제 + (잔여) 잉여-기반 per-round 재선택 부재.**
   - **주범 = age-cap 강제 off-fit.** 기본 cap=100 이 fill 의 ~61%(forced 17,897)를 ideal 무시·강제
     주입 → Σdev 17%. cap 을 키워 강제를 줄이면 Σdev 가 20%→6.3% 로 단조 하락. → **global_age_cap
     이 너무 낮게 설정돼 있었다. Plan 2 에서 knee 재튜닝**(아래 §5).
   - **잔여 ~6.3%p (강제 0 에서도 A 의 1.6%보다 높음) = 진짜 배치방식 차이.** B 는 **잉여가 없어
     머무는 디코더를 매 라운드 못 바꾼다**(빈 슬롯만 재선택). A 는 잉여 10 슬랙으로 **134→62 를 매
     라운드 전부 재선택**해 drift·long개수 변동을 능동 보정한다.
   - → 가설: **C(= A 배치방식 + B 와 동일한 age-cap·캐시)가 강제 동일 조건에서 B 보다 그 잔여(~6%p)
     만큼 Σdev 낮다.** B vs C 가 이 잔여(잉여 per-round 재선택의 고유 이점)를 격리한다.
2. **idle 은 그 자체가 목표가 아니라 TBT 의 한 입력.**
   t_pim 은 Σ decode KV 로 변동 → Σdev↑ 면 t_pim 이 t_gpuA 를 넘겨 **PIM-hiding 이 깨지고** TBT↑.
   동작점에선 idle≈0(PIM 이 GPU-A 밑에 숨음)이라 더 줄일 게 없는데 거기 매달렸다. 진짜로 봐야 할 건
   **TBT·TTFT·SLO goodput**이며, idle 은 TBT 경유로만 이들에 영향한다.

## 3. 세 변종 (동일 워크로드·동일 KPI)

| | 배치 선택 방식 | 글로벌 age-cap | on-node 캐시 |
|---|---|---|---|
| **A** (as-is) | steering 매-라운드 재선택 (잉여 + per-completion 힐링) | ✗ | ✗ |
| **B** (prepo) | 결정론 pre-positioning (고정 set, 재선택 없음) | ✓ | ✓ |
| **C** (신규) | **steering 매-라운드 재선택 (잉여 + 힐링)** | **✓** | **✓** |

- **B vs C — 핵심 격리 실험.** 동일 인프라(글로벌 age-cap + 캐시) 위에서 **배치 선택 방식만**
  다르다. 둘의 Σdev/TBT 차이 = *steering 재선택 vs 결정론 prepositioning* 의 순수 정밀도 차이.
  (이 실험의 의도 = "잉여+힐링 구성 vs 글로벌 정밀 presend, 어느 쪽이 더 정밀한가"를 딱 그것만 본다.)
- **C vs A** — 동일 배치 방식, +글로벌 age-cap +캐시 → 현 설계에 공정성/캐시 더했을 때 효과.
- **C 캐시 용량 = HBM − 가중치 − 활성 2배치 − 잉여(10).** 잉여가 HBM 을 점유하므로 **B 보다 약간
  작다**(잉여 10 vs 124, 차이 소). 하드코딩 금지 — derive 잔여에서 잉여분만 추가 차감.

## 4. 실 KPI (주 지표로 승격, idle/Σdev 는 보조)

- **TBT** = 실현 라운드 시간 = `max(t_pim, t_gpu_a, t_ffn)` (optime, 라운드 composition 기반).
  - `t_pim = t_pim_us(Σ decode KV of μ-batch)` — Σdev 따라 변동.
  - `t_ffn = t_ffn_us(decode_count + prefill_tokens)` · `t_gpu_a = t_gpu_a_us(batch_total, prefill_kv_work_target)` — 거의 상수.
  - idle(자원) = `TBT − 그 자원 시간`.
- **TTFT** = 큐 대기 + 프리필 시간(캐시-aware). **전 요청**(첫 턴 = full prefill / 복귀 턴 = hit/reload/recompute).
- **SLO goodput** = SLO(`TTFT ≤ T_ttft` ∧ `TBT ≤ T_tbt`) 충족 요청의 **출력 토큰수(길이) 가중**
  throughput (tokens/s) + 충족 비율. (표준 LLM SLO-goodput: SLO 내 유효 토큰/초.)
  - 임계 `T_ttft·T_tbt` = balance_time 기반 합리값(예 T_tbt = k×round_us), 스윕 가능.
- **idle-win 카운팅** (라운드마다 셋 계산 후 *가장 오래 idle 한* 자원 = 승):
  - **2자 (GPU-A vs PIM)**: PIM 승률 = **PIM-hiding 빈도**(t_pim ≤ t_gpuA — 대역폭 충돌해도 오버랩되나).
  - **3자 (GPU-A vs PIM vs GPU-B)**: 최장-idle 승 카운트 (관측용).

## 5. 대조 측정 (A vs B vs C, 동일 워크로드)
- **0순위 — global_age_cap 재튜닝.** 기본 100 이 강제 61%로 composition 을 망쳤다(§2). 강제율↔Σdev↔
  캐시 히트↔max_wait knee 를 스윕해 **개선 기본값**을 잡는다(강제 충분히 낮춰 Σdev 회복하되 max_wait
  bounded·캐시 유지). **B 를 개선값으로 재실행해 Plan 1 [REPORT.md](proto/REPORT.md) 의 B·head-to-head
  수치를 갱신**하고 REPORT2 에도 반영.
- 주: **TBT(평균·p99) · TTFT(평균·p99) · SLO goodput(tokens/s·비율)** · PIM-hiding 승률.
- 보조: Σdev(평균·최악) · 캐시 히트율 · forced · max_wait.
- 답할 질문:
  1. **배치 방식만의 정밀도 잔여** (B vs C 의 Σdev·TBT, *동일 age-cap 강제 조건*) — 가설 1 잔여 격리.
  2. **idle → TBT → goodput 사슬이 실재하나** (Σdev 와 TBT/goodput 상관) — 가설 2 검증.
  3. **A 에 글로벌 age-cap·캐시 추가**(=C) 시 max_wait·TTFT·goodput 개선폭.
  4. **PIM-hiding** 이 각 스케줄러에서 얼마나 유지되나(2자 승률).

## 6. 경계
- TBT/TTFT 절대값은 **B200 + Llama70B optime** 종속(HW 바뀌면 재도출). `offload_bw·think_gap·gone_age·
  SLO 임계`는 고정 가정 라벨(스윕 아님 또는 명시 스윕).
- idle/Σdev 는 *목표 아님* — TBT 의 입력으로만 해석. 최종 판단은 SLO goodput.
