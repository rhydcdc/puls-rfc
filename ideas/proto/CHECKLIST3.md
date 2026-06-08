# Plan 3 — 구현 체크리스트 (의존성·컨텐션 + 비용모델 교정)

> 근거 = [../predictive_prepositioning_plan3.md](../predictive_prepositioning_plan3.md). 전부 **C++17**.
> Plan 1·2 산출물 재사용 — 신규 최소. 추가 로직 금지, 누락 금지.

## 0. 제약 (불변)
- [ ] `ideas/proto/` 안에만. `puls-engine/` 등 밖 절대 수정·침범 금지.
- [ ] core 사본 verbatim. 폴더 분리(scheduler/sim/validation).
- [ ] 동작점·op-time 수치 하드코딩 금지(derive/optime).
- [ ] 모듈 단위검증 핵심만(과잉 금지). idle/Σdev 는 보조, 주 = TBT·TTFT·SLO goodput.

## 1. 교정 (작은 변경, 큰 효과)
- [ ] **reload BW 제값**: `harness.h SimConfig.offload_bw_bytes_per_round` 5e9 → **2e7**(SSD ~10 GB/s). CLI 스윕 노출.
- [ ] **콜드스타트 시딩 일치**: `baseline.cpp`·`csched.cpp` 의 콜드스타트를 **큐에서 채우게**(B 처럼) 또는 시드량 일치 → 큐 초기 적체 제거. (prepo.cpp 불변.) 확인: 시작 큐 size 가 B 와 동급(~수백).

## 2. 핵심 — 의존성 + 컨텐션 (P0 계약)
### harness.h — 인스턴스 A 지연 + 컨텐션 헬퍼
- [ ] `instance_a_latency(OpTimes t, double beta)` = `max(t.t_pim, t.t_gpu_a) + beta*max(0, t.t_pim − t.t_gpu_a)`. (조건 `t_pim ≤ t_gpu_a` → 페널티 0.)
- [ ] SimConfig 에 `double contention_beta = 0.5;` (스윕 knob, 0=무컨텐션).

### kpi.h — TBT 식 교체 + 컨텐션 카운트
- [ ] `record_mubatch(..., d, slo, beta)`: TBT = `max(instance_a_latency(t,beta), t.t_ffn) × num_layers`. (β=0 이면 현 max() 와 동일.)
- [ ] 컨텐션 노출 카운트: `t_pim > t_gpu_a` 횟수 + 누적 노출분(`Σ max(0,t_pim−t_gpu_a)×layers`). print 에 노출율·노출시간 추가.
- [ ] PIMwin2/win3 유지(노출율 = 1−PIMwin2 와 정합 확인).

## 3. 드라이버 배선 (3 파일)
- [ ] `baseline.cpp`·`prepo.cpp`·`csched.cpp`: `record_mubatch` 에 `cfg.contention_beta`(또는 CLI argv[7]) 전달. offload_bw 는 SimConfig 교정값/CLI 사용. (메커니즘 불변 — TBT 계산 경로만 갱신.)

## 4. 검증
- [ ] `validation/test_kpi.cpp` 확장: ①`t_pim ≤ t_gpu_a` → 컨텐션 페널티 0, TBT=max(gpua,ffn)×L ②`t_pim > t_gpu_a` → TBT = (max+β·diff vs ffn) 정확 ③β=0 이면 기존 max() 식과 동일(회귀) ④reload 비용 = len×kv_bpt/offload_bw(SSD 값) 정확.

## 5. 스윕 + REPORT3.md
- [ ] 스윕: 글로벌 age-cap · 노드 age-cap · evict_age · **offload_bw** · **컨텐션 β**.
- [ ] **A vs B vs C 진짜-KPI 대조**(TBT 평균·p99 / TTFT / SLO goodput / PIM-노출율 / 컨텐션 페널티시간 | 보조 Σdev/히트/forced/max_wait).
- [ ] **시딩 수정 후 B vs C 히트** = 용량차뿐인지 확인.
- [ ] 사슬 검증: idle(Σdev) → PIM노출 → 컨텐션 → TBT → goodput.
- [ ] 실측만. 경계(HW 종속·가정 라벨) 명시.

## 6. 에이전트 플랜 (총 6, 4 웨이브)
- **P0 — 나(직접):** harness `instance_a_latency`+`contention_beta` · kpi.h TBT 식 교체+컨텐션 카운트 · offload_bw 교정 → 계약 syntax 통과(병렬 고정).
- **P1 — 3 병렬(파일 disjoint):**
  - A1 `baseline.cpp` + `csched.cpp` — 콜드스타트 시딩 일치(§1) + β 배선(§3).
  - A2 `prepo.cpp` — β 배선(§3).
  - A3 `validation/test_kpi.cpp` 확장(§4).
- **P2 — 1:** 스윕(age-cap×2·evict_age·offload_bw·β) 실행 · REPORT3.md(진짜-KPI, 사슬 검증) · 필요시 REPORT2 주석.
- **P3 — 1(검증 전담):** 계획서 대비 감사 — ①전항목 구현 ②추가 로직 0 ③`git status` ideas/ 밖 무변경 ④폴더 분리 ⑤test_kpi+세 드라이버 zero-warning ⑥REPORT3 재현 ⑦β=0 회귀(기존 TBT 와 일치) ⑧누락 0.

총 6 에이전트(3∥1∥1) + P0 직접.
