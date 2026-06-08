# Plan 2 — 구현 체크리스트 (배치방식 격리 + 실 KPI)

> 근거 = [../predictive_prepositioning_plan2.md](../predictive_prepositioning_plan2.md). 전부 **C++17**.
> Plan 1 산출물(`core/`·`scheduler/`·`sim/`·`validation/`) **재사용** — 신규는 최소. 추가 로직 금지, 누락 금지.

## 0. 제약 (불변, Plan 1 동일)
- [ ] 모든 코드 `ideas/proto/` 안에만. `puls-engine/` 등 밖 절대 수정·침범 금지.
- [ ] 외부 core 는 기존 `core/` verbatim 사본 그대로 사용(추가 복사 시 무수정).
- [ ] 폴더 분리: 로직 `scheduler/` · 시뮬 `sim/` · 검증 `validation/`.
- [ ] 동작점·op-time 수치 하드코딩 금지 — derive/optime 산출.
- [ ] 모듈별 단위검증 — 핵심만(과잉 금지).
- [ ] **idle/Σdev 는 보조 지표.** 주 판단 = TBT·TTFT·SLO goodput.

## 1. 재사용 (Plan 1, 무수정)
- [ ] `scheduler/queue.*` (pull_near / pull_slot / max_wait)
- [ ] `scheduler/cache.*` · `scheduler/preposition.*`
- [ ] `sim/workload_mt.*` · `sim/harness.h`
- [ ] `core/*` (optime: `t_pim_us`/`t_ffn_us`/`t_gpu_a_us` — TBT·idle-win 계산에 사용)
- [ ] `sim/baseline.cpp`(A) · `sim/prepo.cpp`(B) — KPI 메트릭 추가만(아래 §3).

## 2. 신규/확장 모듈 (+검증)

### sim/kpi.h — 실 KPI 측정 (header-only, 확장 metrics)
- [ ] **TBT**(라운드별 = `max(t_pim,t_gpu_a,t_ffn)`) 누적: 평균·p99.
- [ ] **TTFT**(전 요청: 첫 턴 full prefill / 복귀 hit·reload·recompute) 누적: 평균·p99.
- [ ] **SLO goodput**: `TTFT≤T_ttft ∧ TBT≤T_tbt` 충족 요청의 **출력토큰(길이) 가중** tokens/s + 충족 비율.
- [ ] **idle-win 카운터**: 2자(GPU-A vs PIM → PIM 승률=hiding) · 3자(GPU-A·PIM·GPU-B) 최장-idle 승.
- [ ] (기존 metrics.h 의 Σdev/캐시/forced/max_wait 는 보조로 병기.)
- [ ] 검증 test_kpi: ①주어진 Σkv 로 TBT=max(셋) 정확 ②t_pim≤t_gpuA 면 2자 PIM 승 ③goodput 이 SLO 임계 경계서 정확히 분기 ④p99 계산 정확.

### sim/harness.h — round_optimes 헬퍼 추가 (확장)
- [ ] `round_optimes(Σdecode_kv, op, model, hw) → {t_pim, t_gpu_a, t_ffn}` (optime 호출, prefill_attn_work = `op.prefill_kv_work_target`).
- [ ] SLO 설정 필드(SimConfig): `T_ttft_us`, `T_tbt_us`(balance_time 기반 기본값).

### sim/csched.cpp — 드라이버 C (신규)
- [ ] **배치 방식 = A 그대로**: 잉여(op.decode_pool) + 2 μ-batch `steer_decode_set` + per-completion ideal=hole 힐링 + 노드 age-cap. (= baseline.cpp 의 노드 루프 재사용.)
- [ ] **+ 글로벌 age-cap**: 힐링 refill 을 `pull_near` → **`pull_slot(ideal,cap,now)`** 로 교체(강제 카운트).
- [ ] **+ 캐시 ON**: eligibility 실값. **용량 = derive 잔여 − 잉여분**(`node_cache_capacity_bytes` 에서 잉여 KV `decode_surplus×ctx_balance×kv_bytes_per_token` 추가 차감) — 하드코딩 금지.
- [ ] 인구 보존 도착 · 복귀 히트 라우팅(tier 크레딧) · warm-gate · KPI(kpi.h) 출력.

## 3. 기존 드라이버 확장 (A·B 에 KPI 추가 — 동일 측정)
- [ ] `baseline.cpp`(A) · `prepo.cpp`(B): 라운드마다 `round_optimes` 로 TBT·idle-win 기록, 전 요청 TTFT, SLO goodput 누적(kpi.h). **A 는 메커니즘 불변**(no age-cap/no cache) — 메트릭만 추가.

## 4. 스윕 + REPORT2.md
- [ ] **0순위 — global_age_cap 재튜닝**: 강제율↔Σdev↔캐시히트↔max_wait knee 스윕(예 100·500·2000·∞) → **개선 기본값** 채택(강제↓·Σdev 회복·max_wait bounded·캐시 유지). 기존 기본 100 은 강제 61%로 과도.
- [ ] **B 를 개선 age-cap 으로 재실행 → Plan 1 [REPORT.md](REPORT.md) 의 B·head-to-head 수치 갱신** (+REPORT2 반영).
- [ ] **A vs B vs C 대조표**(주: TBT 평균·p99 / TTFT 평균·p99 / SLO goodput / PIM-hiding 승률; 보조: Σdev/히트/forced/max_wait).
- [ ] **B vs C Σdev·TBT 격리** — *동일 age-cap 강제 조건*에서 잔여(잉여 per-round 재선택 부재) gap 측정 — 가설 1 잔여.
- [ ] **Σdev ↔ TBT ↔ goodput 상관**(idle 사슬) — 가설 2.
- [ ] SLO 임계(T_ttft·T_tbt) 스윕 → goodput knee.
- [ ] 실측 숫자만. 경계(HW 종속·가정 라벨) 명시.

## 5. 에이전트 플랜 (총 6, 4 웨이브)
- **P0 — 나(직접):** `sim/kpi.h`(KPI 측정 계약) + `harness.h` round_optimes/SLO 필드 확정 → 계약 syntax 통과(병렬 고정).
- **P1 — 3 병렬(파일 disjoint):**
  - A1 `sim/csched.cpp` (드라이버 C) — 빌드·sane run.
  - A2 `baseline.cpp` + `prepo.cpp` 에 KPI 메트릭 추가(두 파일, 한 에이전트) — 빌드·sane run.
  - A3 `validation/test_kpi.cpp` (kpi.h 검증 4체크) — 통과.
- **P2 — 1:** 스윕 실행(A·B·C + SLO 임계) · REPORT2.md(실측, 가설 1·2 답).
- **P3 — 1(검증 전담):** 계획서 대비 감사 — ①전항목 구현 ②추가 로직 0 ③`git status` ideas/ 밖 무변경 ④폴더 분리 ⑤test_kpi + 세 드라이버 zero-warning 통과 ⑥REPORT2 수치 재현 ⑦누락 0.

총 6 에이전트(3∥1∥1) + P0 직접.
