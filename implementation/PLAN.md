# PULS Scheduler — Implementation Plan

**Working draft (2026-05-26).** References: [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §5–§6, [`../README.md`](../README.md) Phase 1.

본 문서는 README Phase 1 / ARCHITECTURE §5.5 · §6.7 의 self-authored scheduler framework 의 구현 계획 세부서. Module 분해 + phase 별 *Implementation* (구현 산출) + *Unit Tests* (단위별 검증) + *Acceptance* (phase 진입 조건) 의 항목화.

---

## Deliverables (Scope Anchor)

본 RFC 의 산출은 **두 가지로 한정**:

**D1. 동작하는 scheduler.** Event-driven DAG dispatch + adaptive admission + Instance A/B pipeline 이 임의 ctx × batch trace 위에서 *처리 시간 균형 잡힌 μ-batch 구성*으로 수렴. 검증 = §6.5 dispatch trace emergence + admission deadband 수렴 + F4 steady-state + invariant 0 위반 + determinism. → **Impl-9 acceptance (시뮬레이터 통과 조건)**.

**D2. F1~F5 가속 source decomposition.** ARCHITECTURE §5.7 의 F1·F2·F3·F5 각 source 의 isolated cycle ratio 를 workload regime (ctx × batch) 격자 위에서 산출. F4 는 D1 의 전제 검증. → **Impl-10 (Phase 3 calibrated projection — GPU 실측 + PIM Ramulator2 추정)**.

**산출하지 않음 (scope 외):**

- Comparative baseline reimplementation (Sarathi-Serve · vLLM) 과의 배수 비교 — 사유: (i) pre-HW dummy time 위 reproduction 정합 판단 불가, (ii) Phase 3 후에도 PULS silicon 부재로 양방 실측 비교 불가능.
- 절대 throughput · TTFT · TPOT · goodput · SLO per GPU — 사유: silicon 부재.
- Silicon-validated PULS measurement — 사유: PULS fab 없음. Phase 3 후에도 PIM side 는 estimate (Ramulator2 + JEDEC spec scaling, 출처 라벨 동반).

D1 과 D2 의 관계: D1 (배치 균형) 이 *전제* — staggering steady-state 가 안 잡히면 F2/F3 의 `max()` ratio 자체가 성립 안 함. D2 (가속 배수) 가 *산출* — 균형 위에서 각 source 가 얼마나 기여하는지의 분해.

---

## 0. Completeness Definition

스케줄러의 *완전한 module list* 는 사전 정의 불가. 이유:

- **Canonical reference 부재** — LLM serving scheduler 의 표준 module 정의 없음. vLLM · Sarathi-Serve · SGLang 각자 다른 abstraction.
- **Edge case 는 구현 중 노출** — 첫 trace replay 시점에서 누락이 발견되는 *discovery* 영역.
- **추상화 경계 모호** — clock 을 별도 모듈로 둘지 EventQueue 내부 state 로 둘지 같은 결정은 구현 디테일.

따라서 본 plan 은 *시작점* 이며, 완전성의 정의를 다음 acceptance test 로 환원.

### Completeness Acceptance Test (End-to-end Runnable Criterion)

scheduler 가 **완성** 의 시점은 다음 5 조건 동시 충족:

- **C1.** Synthetic 100-request trace 입력 → 모든 request 가 EOS 또는 max_tokens 도달로 종료
- **C2.** Schema-valid 한 *구조적 산출* — §6.5 dispatch trace (Init/T1–T5) 재현 + adaptive admission deadband 수렴 trace + F1·F2·F3·F5 ablation cycle ratio 표. *Comparative baseline 미산출 (Deliverables 정합).*
- **C3.** KV slot 누수 없음 — completion 후 capacity 회수 정상
- **C4.** Invariant (I1–I5) 위반 0 회 over full trace
- **C5.** Determinism — 동일 seed + trace → bit-exact 구조 산출

본 5 조건은 Impl-9 (End-to-end Driver) 의 acceptance 와 동일. 이 시점 이후의 *추가* gap 은 plan 갱신으로 흡수 (iterative refinement). 사전에 모든 module 을 정의하려는 시도는 yagni · 과잉 abstraction risk.

---

## 0.5 Implementation Principles

### Numeric Value Policy (Impl-1~9)

Impl-1~9 는 로직 · 알고리즘 · 자료구조 · invariant 강제 의 구현 영역. 정량 수치 의존 금지.

- **금지** — Phase 0 산출 추정값 의 직접 사용 / 하드코딩 / acceptance 기준화:
  - η_HBM (HBM 실효 대역폭 비율)
  - NVLink 실측 BW
  - KV length variance 통계값
  - FFN N_sat (saturating knee)
  - **Ramulator2 산출 HBM4 tile time** — Ramulator2 에 HBM4 모델 부재로 JEDEC spec timing param + HBM3 보수적 인용 + 자체 모델링으로 산출. 실측 silicon 까지 *추정* 카테고리.
- **예외 — RTL FSM cycle 단독.** Yosys + ASAP7 + OpenSTA pre-CTS flow 로 합성 확정. 회로가 변하지 않으면 cycle 도 변하지 않음 → config 하드코딩 허용.
- **모든 정량값** — `config` 의 placeholder field 참조. 로직 코드는 변수명 lookup 만 사용 (magic number 금지).
- **Dev / test 단계 placeholder** — 임의 dummy 값. 자료구조 동작 + 구조적 property 검증 전용. 예: `config.pim_tile_time = {FP8: 1.0, FP16: 2.0}` (절대값 무의미, ratio 2× property 만 보존).
- **실측 값 / Ramulator2 산출값 주입** — Impl-10 / Phase 3 영역. 본 영역의 로직 코드 변경 0 — config field 만 교체.
- **출처 라벨링** — Impl-10 이후 placeholder 에 값이 주입될 때 출처 label (예: `ramulator2_hbm4_estimated_jedec_spec` · `nvlink4_sxm_spec` · `lab_blackwell_measured`) 을 동반. `eval` 출력에 함께 기록 → 정량 metric 의 estimation 출처 disclosure.

### Test Property Discipline

Unit test 는 *구조적 properties* 만 검증 — 정량 일치 (예: "±5%") 검증 금지.

- ✓ "동일 dummy 입력 → 동일 출력" (determinism)
- ✓ "입력 monotonic 증가 → 출력 monotonic 증가" (예: KV 길이 ↑ → PIM op time ↑)
- ✓ Invariant (I1–I5) 위반 0
- ✓ State transition 단방향성
- ✓ Lookup 정합 (출력 = config placeholder 값)
- ✗ "Phase 0 측정값 ± 5%" — Impl-10 으로 deferred

### Scheduler 의 Tile Time 의존 (PULS 고유 logic 영역)

Production scheduler (vLLM · Sarathi-Serve) 에 reference 없음. PIM tile time 은 scheduler 의 의사결정 input:

- **§3.5.2 Computed Wait** — `t_end = t_start + num_tiles × tile_time` (interrupt 없는 substrate 의 직접적 귀결)
- **§6.3 DAG dispatch** — PIM 완료 시각 사전 계산으로 다음 GPU op pre-scheduling
- **§5.1 / §5.6 / §6.4** — `kTotalDecider.solve`, double-buffering balance, adaptive admission 의 `t_PIM` 평가에 tile time 필요

따라서 `PIMExecutor.tile_time(regime)` 은 단순 op time 산출용을 넘어 scheduler logic 의 일부. **단 그 *값* 은 config placeholder lookup, *로직* 만 Impl-4 / Impl-3 / Impl-2 에 분산.** 본 정책으로 logic 과 numeric 의 분리 유지.

### Test 신뢰도 보강 여지 (Impl-N 별 reminder)

각 Impl 의 §4 *Unit Tests* 체크리스트는 *시작점*. "테스트 잘못 통과" 시나리오 차단을 위해 각 Impl 진입 시점에 다음 카테고리 보강 검토 — meta-test (plan 정합 자동 검증) · negative parametrized (invalid case cross-product) · cross-module invariant · self-review (commit 직전 mechanical 점검).

- **Impl-1** — 자료구조 / state machine. Meta-test (모듈 inventory · enum 완전성 · DAG precedence) + negative parametrized (invalid transition cross-product) + cross-module invariant (window↔DAG · add↔remove round-trip · frozen immutability)
- **Impl-2** — Invariant 강제 + DAG dispatcher. Negative parametrized (I1·I2·I3·I4·I5 violation 각각) + §6.5 dispatch trace Init/T1–T5 sequence meta-test + priority dequeue boundary parametrize
- **Impl-3** — Admission + k_total. Boundary parametrize (N == N_sat ± 1 · k_total 9-step 전수) + KV capacity round-trip + idle telemetry reproducibility
- **Impl-4** — PIM executor. Regime 분기 cross-product (FP8 · FP16) + FSM determinism N-run repeat + lookup 정합 (`tile_time` == `config.pim_tile_time[regime]`)
- **Impl-5** — Instance A/B pipeline. Handoff shape validation (decode · prefill 양 case fixed-shape 강제) + L-layer iteration meta-count + steady-state cycle = max(A, B) round-trip
- **Impl-6** — Trace replay + completion. KV slot 회수 round-trip (admit ↔ release) + trace replay multi-seed determinism + completion 검출 boundary
- **Impl-7** — (Removed) Comparative baseline reimplementation scope 외 (Deliverables 정합)
- **Impl-8** — Evaluator. Dispatch trace 캡처 + admission 수렴 trace + F1~F5 cycle ratio decomposition schema 정합. *절대 metric (TTFT · TPOT) 미산출 (Deliverables 정합)*
- **Impl-9** — End-to-end driver. §0 C1–C5 acceptance 각각 별도 test + full trace replay bit-exact determinism
- **Impl-10** — Sensitivity sweep. Sweep grid coverage meta-test (ctx × batch 누락 0) + ablation flag round-trip + F1·F2·F3·F5 isolation 검증

각 Impl 의 plan 문서 (`implementation/plans/impl_N.md`, local-only) 에서 위 reminder 를 출발점으로 구체 test 항목 확장. Self-review (commit 직전 mechanical 검증 — naming grep · import 순환 · LOC ceiling · plan diff · CLAUDE.md §2 sanity) 는 모든 Impl 공통.

### MODULES.md 갱신 (Impl-N 별 공통 reminder)

각 Impl 의 commit 직전 (Self-review 동시 영역) 에 `implementation/MODULES.md` 의 해당 Impl 섹션을 갱신 — 그 Impl 에서 신설 또는 변경된 모듈의 한 줄 역할 설명 추가/수정. 신설 모듈이 없는 Impl (skeleton 에 logic 만 채우는 경우) 도 기존 모듈의 한 줄 설명 갱신 필수 (예: Impl-2 의 `main_loop.py` 는 skeleton → 실 dispatcher 로 의미 변경, `dispatcher.py` 신설 — MODULES.md 에 반영).

### Pre-HW 산출 영역 정의 (Deliverables 정합)

Impl-1~9 (dummy time model 위) 가 산출 *가능 / 불가능 / 스코프 외* 항목:

**가능 (research artifact, D1 검증 자산):**

- §6.5 dispatch trace emergence (back-fill at T3 등 — priority dequeue + DAG 자연 산출)
- Adaptive admission deadband 수렴/발산 trace (control loop 안정성)
- F1·F2·F3·F5 ablation 시 cycle 식 ratio 변화 패턴 (구조 검증)
- F4 steady-state 도달 (μ-batch staggering 자기-동기화)
- Invariant I1~I5 위반 0, determinism, KV 누수 0 (engineering hygiene)

**불가능 (Impl-10 calibrated projection 후에도, silicon 부재):**

- TTFT · TPOT · throughput · goodput · SLO per GPU 절대값
- Silicon-validated PULS measurement

**스코프 외 (애초에 산출 시도 안 함, Deliverables 정합):**

- ~~PULS vs Sarathi/vLLM 배수 비교~~ — Comparative baseline reimplementation 본 RFC scope 외
- 사유: (i) pre-HW dummy time 위 reproduction 정합 판단 기준 부재, (ii) Phase 3 후에도 PULS silicon 부재로 양방 실측 비교 불가능

본 inventory 가 D1 (Impl-9 시뮬레이터 통과) 과 D2 (Impl-10 F1~F5 decomposition) 의 검증 자산 한도를 lock-in.

---

## 1. Scope

### 1.1 In Scope

- μ-batch composition + token-level admission control
- Event-driven DAG dispatcher (§6.3)
- Adaptive admission with hysteresis deadband (§6.4)
- k_total decision (PIM-side dial, §5.1)
- Instance A/B inter-instance pipeline dispatch (§3.4 · §5.2)
- PIM executor emulator (Ramulator2 cycle wrapping)
- Trace replayer + structural evaluator (dispatch trace · 수렴 trace · F1~F5 ratio)
- End-to-end driver + request lifecycle + KV accounting (§0 완전성 충족용)

### 1.2 Out of Scope

- **Comparative baseline scheduler reimplementation (Sarathi-Serve · vLLM).** 사유: (i) pre-HW dummy time 위 reproduction 정합 판단 기준 부재, (ii) Phase 3 calibrated projection 후에도 PULS silicon 부재로 양방 실측 비교 불가능, (iii) RFC deliverable 을 *D1 (동작하는 scheduler) + D2 (F1~F5 source decomposition)* 으로 한정 — comparative axis 미포함 (Deliverables 정합).
- **절대 metric (TTFT · TPOT · throughput · goodput · SLO per GPU).** 사유: silicon 부재로 anchored absolute time 산출 불가.
- KV page → 물리 채널 매핑 (page allocator 영역)
- Channel-row sharding 정책 결정 (deploy-time architectural decision, §3.3)
- Kernel launch · CUDA stream · weight streaming (model executor + driver)
- HBM transaction · TSV 점유 · PIM_toggle bit 발사 (hardware)
- Real-hardware jitter modeling (σ measurement; §6.4 future work)
- Preemption / CPU swap / multi-tenant / speculative · prefix cache hook (v1 보류)

---

## 2. Module Decomposition

| Module | 책임 | Reference |
|---|---|---|
| `config` | Global parameters — model (L · hidden · num_heads · GQA · head_dim) · HW (8 GPU TP · 8 stack × 32 channel · FP8 KV) · SLO · workload · seed | §3.4 Case A |
| `clock` | Virtual simulation time — single source-of-truth, EventQueue 와 동기 | §3.5.2 · §6.3 (computed wait 기반) |
| `request` | Request 자료구조 + state machine (pending → prefill → decode → completed) | (production scheduler convention) |
| `request_queue` | Waiting queue between trace arrival and admission; FIFO 기본 | (production scheduler convention) |
| `kv_accountant` | Aggregate KV slot capacity 추적 — admit 시 demand 검증, completion 시 회수 | §3.3 (aggregate Instance A HBM capacity) |
| `mubatch` | μ-batch 자료구조 — 요청 집합, prefill chunk + decode 토큰 분리, weight 공유 invariant | §6.1 |
| `dag` | Dependency DAG — 4 노드/μ-batch + I1·I2·I3 엣지 자동 생성 | §6.3 |
| `invariants` | I1–I5 강제 — correctness gate + resource exclusivity gate | §6.2 |
| `dispatcher` | Event-driven main loop — priority dequeue, GPU·PIM 큐 separate | §6.3 |
| `admission` | Layer 1 admission — μ-batch composition + idle telemetry + deadband | §6.4 |
| `k_total` | `t_proj` · `t_PIM(k_total)` → k_total 결정 (aggregate 9-step dial) | §5.1 · §5.6 |
| `pim_emulator` | Ramulator2 cycle → blocking event; FP8/FP16 regime branching; SP-PIM 2048 channel cooperation | §3.1 · §3.4 |
| `gpu_executor` | Op time predictor + kernel completion event 발생 | §6.7 |
| `instance` | Instance A/B 분리 dispatch + NVLink handoff + fixed-shape `[B×hidden]` / `[(B·chunk)×hidden]` | §3.4 · §5.2 |
| `forward_pass` | L-layer iteration loop per token; layer_state 관리 | §3.4 (L × cycle = forward pass) |
| `completion` | EOS / max_tokens 검출 + KV slot 회수 + completion timestamp 기록 | (production scheduler convention) |
| `trace` | Long-ctx production trace + 1M-class benchmark dataset feed | §8 |
| `eval` | Dispatch trace 캡처 + admission 수렴 trace + F1~F5 cycle ratio decomposition + idle fraction + PIM utilization. *절대 metric (TTFT · TPOT · throughput · goodput) 미산출 (Deliverables 정합)* | §6.7 |
| `run` | Outer driver loop · initialization · 모듈 wiring · termination | (entry point) |

구현 언어는 Python 권장 — production scheduler reference (vLLM · Sarathi-Serve) 와 정합. 최종 결정은 Impl-1 진입 시.

**Note.** 위 module list 는 *시작점* 이며 §0 정합. Impl-9 acceptance test 실행 시 추가 발견되는 누락은 plan 갱신으로 흡수.

---

## 3. Time Model Components

| 항목 | Config field (Impl-1~9 dummy) | 실측 / 추정값 주입 시점 |
|---|---|---|
| GPU op time | `config.gpu_op_time[op_type]` (dummy: 1.0 μs) | Impl-10 (랩실 블랙웰 8 GPU 실측) |
| PIM decode-attn tile time | `config.pim_tile_time[regime]` (dummy: FP8=1.0, FP16=2.0 — ratio 2× property 만 보존) | Impl-10 (Ramulator2 HBM4 추정 cycle ingest, `ramulator2_hbm4_estimated_jedec_spec` 라벨) |
| NVLink handoff | `config.nvlink_time_per_byte` (dummy: 1.0 ns/B) | Impl-10 (NVLink 4 SXM 실측 또는 spec 인용) |
| HBM4 BW (참고) | `config.hbm4_bw` (dummy: 1.0 GB/s) | Impl-10 (Ramulator2 추정) |
| **RTL FSM cycle (예외)** | **`config.rtl_fsm_cycle_per_tile` — 합성 확정값 하드코딩 OK** | **(주입 완료, 변동 없음)** |

§0.5 Numeric Value Policy 정합. Impl-1~9 의 모든 로직 코드는 위 config field 의 *변수명 lookup* 만 사용. Dummy 값은 자료구조 + 구조적 property 검증 전용 — 정량 평가 metric (가속 배수 · TTFT · TPOT 절대값) 은 Impl-10 / Phase 3 영역까지 의미 없음. 각 placeholder 의 출처 라벨은 `eval` 출력에 함께 기록 (estimation 출처 disclosure).

---

## 4. Implementation Phases (Checklist)

각 phase 는 *Implementation* + *Unit Tests* + *Acceptance* 의 3 단 구성. **Impl-9 가 end-to-end runnable 의 검증 phase** — §0 완전성 정의의 실체.

### Impl-1 — Core Data Structures + Event Loop + Foundations ✓ (commit `943eca5`)

**Implementation:**
- [x] `Config` — model params · HW params · SLO targets · workload params · seed (dataclass / 단일 dict)
- [x] `Clock` — virtual simulation time; current_time getter + advance(dt); EventQueue 의 source-of-truth
- [x] `Request` — id · prompt tokens · decoded tokens · KV length · arrival time · SLO target · **state field (pending / prefill / decode / completed)**
- [x] `RequestState` — state transition (pending → prefill → decode → completed) + invalid transition reject
- [x] `MicroBatch` — request set · prefill chunk allocation · decode token list · layer index
- [x] `Node` — node type (QKV / prefill-attn / decode-attn / O-proj) · μ-batch ref · state (pending → ready → running → done)
- [x] `DAG` — 4 노드/μ-batch 자동 생성 + I1·I2·I3 precedence edge 자동 생성
- [x] `Event` — event type (kernel-completion / request-arrival / admission-tick) · timestamp · payload
- [x] `EventQueue` — time-ordered priority queue, Clock 과 동기
- [x] `InFlightWindow` — 3 μ-batch sliding window; μ-batch 추가/제거 시 DAG 동기 갱신
- [x] Main loop skeleton — event pop → invariant check → dispatch → state update (skeleton; `_handle` body 는 Impl-2 영역)

**Unit Tests:** (총 60 passed; PLAN §0.5 보강 9 항목 포함 — `implementation/plans/impl_1.md` 참조)
- [x] `Config` — required field 누락 시 fail-fast; seed 변경 시 RNG 영향 검증
- [x] `Clock` — current_time monotonic non-decreasing; advance(dt) 후 정확 시간 갱신
- [x] `Request` 생성 + field round-trip 검증
- [x] `RequestState` — pending → prefill → decode → completed 정상 진행; 역방향 transition reject
- [x] `MicroBatch` — prefill chunk + decode token split 정확성 (token mix invariant)
- [x] `Node` — state transition 단방향성 (역방향 transition 거부)
- [x] `DAG` — 단일 μ-batch 위 4 node + 4 precedence edge (I1·I2·I3) 자동 생성
- [x] `EventQueue` — timestamp 오름차순 pop, tie-break by insertion order; Clock 과 동기
- [x] `InFlightWindow` — 3 μ-batch 회전; 4 번째 μ-batch admit 시 oldest 자동 eviction
- [x] End-to-end smoke — synthetic 1 μ-batch event 가 dispatch 까지 도달

**Acceptance:** ✓ Synthetic 10 μ-batch trace 위에서 event queue 가 시간순 dequeue, DAG 자동 생성, sliding window 가 무한 메모리 누적 없이 회전.

---

### Impl-2 — Invariants + DAG Dispatcher ✓ (commit `a09a2c1`)

**Implementation:**
- [x] `Invariants.check_I1` — prefill-attn(X) 가 QKV(X) 완료 후만 ready
- [x] `Invariants.check_I2` — decode-attn(X) 가 QKV(X) 완료 후만 ready
- [x] `Invariants.check_I3` — O-proj(X) 가 prefill-attn(X) ∧ decode-attn(X) 둘 다 완료 후만 ready
- [x] `Invariants.check_I4` — 시점 t 에 GPU GEMM/attention op 1 개만 active
- [x] `Invariants.check_I5` — 시점 t 에 PIM decode-attn op 1 개만 active (multi-head · multi-request batching 은 op 내부 자유)
- [x] `Dispatcher.refresh_ready` — 모든 precedence done 인 미실행 노드 집합 산출
- [x] `Dispatcher.pick_gpu` — priority dequeue: O-proj > prefill-attn > QKV. Tie-break: oldest μ-batch first
- [x] `Dispatcher.pick_pim` — PIM ready set 에서 oldest μ-batch
- [x] `Dispatcher.dispatch_gpu` / `dispatch_pim` — executor 호출 + node state → running
- [x] Look-ahead / back-fill emergence — 명시 정책 없이 priority dequeue 의 자연 산출 (§6.5 trace 정합)

**Unit Tests:** (총 47 신규 passed; 기존 60 + 신규 47 = 107 — `implementation/plans/impl_2.md` 참조)
- [x] I1 — QKV(X) done = False 상태에서 prefill-attn(X) dispatch 시도 시 차단 (PENDING/READY/RUNNING 3 case parametrize)
- [x] I2 — QKV(X) done = False 상태에서 decode-attn(X) dispatch 시도 시 차단 (3 case parametrize)
- [x] I3 — prefill 또는 decode 둘 중 하나만 done 일 때 O-proj 차단; 둘 다 done 일 때만 ready (prefill 미완 3 + decode 미완 3 + both 미완 1 case)
- [x] I4 — GPU 활성 op 1 개 상태에서 두 번째 GPU op dispatch 시도 시 차단
- [x] I5 — PIM 활성 op 1 개 상태에서 두 번째 PIM decode-attn dispatch 시도 시 차단
- [x] Priority — O-proj / prefill-attn / QKV 모두 ready 일 때 O-proj 선택 + boundary parametrize (O-proj 부재 → prefill, prefill 부재 → QKV, all 부재 → None)
- [x] Tie-break — 동일 priority 다중 μ-batch ready 일 때 oldest μ-batch 선택 (GPU·PIM 양 case)
- [x] §6.5 trace fixture — {P, M, N} 3 μ-batch 구성 + deterministic op time (PIM=3.0 > GPU=1.0, ratio property 보존) → Init / T1–T5 dispatch 시퀀스 bit-exact 재현

**Acceptance:** ✓ I1–I5 위반 0 회 over 100 μ-batch synthetic stress trace. ✓ §6.5 dispatch trace (Init / T1–T5) 재현.

---

### Impl-3 — Admission Controller + k_total Decision + Request Queue + KV Accounting

**Implementation:**
- [x] `RequestQueue` — waiting queue; FIFO admission; bounded queue overflow policy (reject 또는 backpressure)
- [x] `KVAccountant` — aggregate KV slot capacity 추적; `admit(req)` 시 `kv_demand ≤ remaining` 검증; completion 시 `release(req)` 회수
- [x] `IdleTelemetry` — GPU · PIM idle fraction per-iteration 누적; instance A · B 별 분리 측정 *(Impl-3: 단일 instance; A·B 분리는 Impl-5)*
- [x] `Deadband` — ctx-tiered static lookup (short 2k–8k · mid ~32k · long 128k–1M; §6.4 표 정합)
- [x] `Admission.layer1` — μ-batch composition 결정: prefill chunk 토큰 수, decode batch size, N. Candidate pool = `RequestQueue`. 자원 제약 = `KVAccountant`
- [x] `Admission.mfu_floor` — `N ≥ N_sat` (FFN GEMM saturating knee) 강제
- [x] `Admission.balance_inter_AB` — `A_cycle` vs `B_cycle` 차이 기반 prefill chunk admit 조정
- [x] `Admission.balance_intra_A` — GPU vs PIM idle fraction 기반 decode / prefill chunk admit 조정
- [x] `kTotalDecider.solve(t_proj, t_PIM_fn, N_decode)` — `max k_total s.t. t_PIM(k_total, N_decode) ≤ t_proj`; **k_total ∈ {0, 256, 512, ..., 2048}** (aggregate, 8 GPU lock-step 전제, 9-step dial); per-GPU `n × 32, n ∈ {0..8}` → aggregate `k_total = 8 × n × 32`
- [x] `kTotalDecider.over_budget_handler` — `t_PIM(2048) > t_proj` 진입 시 admission layer 1 escalation *(KTotalResult.over_budget flag 로 caller 에 escalation signal 전달; admission.layer1 이 spec.over_budget 으로 propagate. 자발적 escalation loop 은 Impl-9 driver 영역)*

**Unit Tests:** (총 114 신규 passed; 기존 107 + 신규 114 = 221 — `implementation/plans/impl_3.md` 참조)
- [x] `RequestQueue` — FIFO 동작; bounded queue overflow handling (overflow 시 reject 또는 backpressure signal)
- [x] `KVAccountant` — capacity 초과 admit 시 reject; completion 후 `release` 호출 시 capacity 정상 회수
- [x] `IdleTelemetry` — 합성 event sequence (활성 / idle 구간 명시) 위에서 idle fraction 계산값 정확
- [x] `Deadband` — 각 ctx tier (short / mid / long) lookup 의 deterministic 출력
- [x] `Admission.layer1` — 동일 input → 동일 N · chunk size 출력 (reproducibility)
- [x] `Admission.mfu_floor` — `N < N_sat` 입력 시 N_sat 으로 clamp 또는 reject
- [x] `Admission.balance_inter_AB` — A_cycle > B_cycle 시 prefill chunk admit 증가 방향
- [x] `Admission.balance_intra_A` — GPU idle > θ_high 시 decode admit 증가; PIM idle > θ_high 시 prefill chunk admit 증가
- [x] `kTotalDecider.solve` — 9-step k_total 영역에서 max k 선택 정확성; 동일 입력 → 동일 출력
- [x] `kTotalDecider.over_budget_handler` — `t_PIM(2048) > t_proj` 시 escalation signal 발사 검증
- [x] Stack-granularity 검증 — `k_total mod 256 = 0` (per-GPU n × 32 with same n across 8 GPUs)

**Acceptance:** ✓ Balanced regime 에서 idle fraction 이 deadband 내 oscillation 없이 수렴 (multi-iter convergence test). ✓ `kTotalDecider` 출력 결정론 (1000-call bit-exact). ✓ KV capacity overflow / underflow 0 회 (단독 stress + admission↔mock completion cross-module roundtrip).

---

### Impl-4 — PIM Executor Emulator ✓ (commit pending)

> **Signature divergence note.** PLAN 초기 signature `op_time(k_channels, N_decode, N_kv_avg)` 가 ARCH §3.1 (FSM cycle structure invariant) · §3.4 (KV-row sharding exact) 정확 반영 후 `op_time(k_channels, kv_rows_total)` 로 갱신 (§0 iterative discovery 정합). `dispatch` 메서드는 ARCH §3.5.2 (no separate synchronization mechanism) 정합으로 *전면 미구현* — PIMExecutor 는 stateless 시간 계산기, dispatch agent 아님 (dispatcher 가 직접 KERNEL_COMPLETION event push).

**Implementation:**
- [x] `PIMExecutor.tile_time()` — `config.model.kv_precision` 기반 regime lookup (FP8 / FP16). System-wide 결정 (ARCH §3.1 정합)
- [x] `PIMExecutor.op_time(k_channels, kv_rows_total)` — SP-PIM aggregate (Q-replicate + KV-row sharding). 단일 ceil 산식 `ceil(kv_rows_total / (k × rtl_fsm_tile_rows)) × tile_time + broadcast` (ARCH §3.4 정합, Hermite identity 위 두 단계 ceil 과 수학적 등가). batch dim 무관 (ARCH §3.1 FSM cycle invariance)
- [x] Ramulator2 cycle 데이터 loader (JSON minimal schema) — staticmethod `load_ramulator2_cycles(path)`. 5 malformed case fail-fast. **실 Ramulator2 HBM4 cycle ingest 는 Impl-10 영역. Impl-4 는 fixture dummy + schema 안전성만 (§0.5 정합)**
- [x] SP-PIM cross-GPU lock-step cooperation overhead model — Binary `k > k_per_gpu_max` (= 256, ARCH §3.2 literal) 시 `pim_broadcast_latency_ns_cross_gpu` 가산. **값은 config placeholder dummy**
- [x] FSM determinism — jitter ±0, pure function (ARCH §3.5.2 Computed Wait)
- [x] `dispatcher.py` PIM branch — `pim_executor.op_time(k=k_max, rows=tile_rows)` placeholder default args 형식 wiring (진짜 signal flow 는 Impl-5)

**Unit Tests:** (총 94 신규 passed; 기존 221 + 신규 94 = 315 — `implementation/plans/impl_4.md` 참조)
- [x] `tile_time` — FP8 / FP16 regime cross-product lookup 정합 (`pim_executor_fp16` fixture 위 dataclasses.replace) + ratio 2× property (ARCH §6.6 placeholder)
- [x] `op_time` — k_channels sweep monotonic 비증가 (cross-GPU 영역 broadcast 분리) + kv_rows_total sweep monotonic 비감소 + KV-row sharding ratio property (k 2× → per_channel 1/2)
- [x] `op_time` — batch dim invariance (1) signature inspection (N_decode 부재 구조 강제) + (2) 행동 sweep parametrize (n_decode ∈ {1, 8, 64, 256, 1024, 4096}) bit-exact 동일 + (3) Hermite identity 100 random sample 위 단일 ceil ↔ 두 단계 ceil 수학 등가
- [x] Broadcast overhead — k=256 vs k=257 boundary exact diff == `pim_broadcast_latency_ns_cross_gpu` + cross-GPU 영역 (512, 1024, 1536, 2048) constant + single-GPU 영역 (1, 32, 64, 128, 256) zero
- [x] Ramulator2 loader — valid round-trip + 5 malformed (empty / non-list / missing field / wrong type regime · tile_cycle / negative cycle / duplicate regime / file not found / non-dict entry) fail-fast
- [x] FSM determinism — 1000-call bit-exact + multi-instance bit-exact + RNG independence
- [x] ARCH literal — §3.1 (tile_rows=32) · §3.2 (256 per-GPU) · §3.4 (2048 aggregate) · §6.6 (ratio 2×) · §5.1 (k=0 → 0.0)
- [x] Cross-module: `k_total.solve` + real PIMExecutor.op_time bind (closure injection) — max feasible + only k=0 feasible + monotonic in kv_rows + dial stack-granularity + determinism 1000-call + **n_decode sweep invariance (Q12 cross-module 행동 검증)**
- [x] Cross-module: dispatcher PIM branch wiring — `_op_time` ↔ `pim_executor.op_time` bit-exact + ARCH §3.5.2 Computed Wait timestamp 정합
- [x] Meta-test signature divergence (PLAN-code 정합 lock-in) — op_time params bit-exact `{self, k_channels, kv_rows_total}` + no regime arg + no dispatch method + no clock/queue field
- [x] Meta-test PLAN literal — `_EXPECTED_MODULES` 에 `pim_emulator` + ModelConfig.kv_precision · TimeConfig.rtl_fsm_tile_rows · pim_broadcast_latency_ns_cross_gpu 필드 정합 + PIMExecutor public method inventory bit-exact + pim_tile_time_ns 양 regime key

**Acceptance:** ✓ FP8 / FP16 regime 분기 결정론 (kv_precision swap 정합). ✓ Tile time `==` config bit-exact (lookup 정합). ✓ FSM determinism 1000-call bit-exact + multi-instance. ✓ Cross-module integration (k_total ↔ PIMExecutor real injection). ✓ ARCH literal lock-in (§3.1 · §3.2 · §3.4 · §3.5.2 · §5.4 · §6.6). **정량 fidelity (Ramulator2 추정값 정합) 는 Impl-10 / Phase 3 영역 deferred (§0.5 정합).**

---

### Impl-5 — Instance A/B Pipeline + Forward Pass + Inter-instance Handoff ✓ (commit pending)

**Implementation:**
- [x] `Instance` 클래스 — GPU pool + (Instance A 한정) PIM pool. TP=8 lock-step 위 GPU = 단일 자원 (Q2)
- [x] `InstancePipeline` — 단일 layer A → B 구조 + handoff (L-loop 은 forward_pass 책임, Q3)
- [x] `NVLinkTransfer.time(tensor_shape)` — `config.nvlink_time_per_byte × bytes(tensor_shape)` pure function. decode `[B × hidden]` + uniform-chunk prefill `[(B · chunk) × hidden]` 양 case 지원. Event push · 자원 lock 안 함 (Q4). **계수 자체는 config placeholder (Impl-10 에서 실측 / spec 인용으로 교체). §5.2 정합.**
- [x] Async transfer hiding — `steady_state_cycle = max(A_cycle, B_cycle)` ARCH literal 산식 위 자연 흡수 (Q7 — NVLink event 별도 push 안 함)
- [x] Steady-state pipeline cycle measurement = `max(A_cycle, B_cycle)` (runtime getter, Q6)
- [x] Fixed-shape handoff 강제 — Instance B 가 ragged batching 미수신 (§5.2). Violation 시 raise (Q5)
- [x] `ForwardPass.run(μ_batch)` — L-layer iteration (LayerState.advance 의 L 회 반복 + token decode signal trigger). 실 instance_pipeline.dispatch 통합은 Impl-9 영역
- [x] `LayerState` — μ-batch 의 current_layer_index 추적; L 도달 시 token decode signal 발사 (advance True 반환)
- [x] `MicroBatch` 의 `k_total` · `kv_rows_total` · `current_layer_index` 3 필드 신설 (Impl-4 carry-over O4.1 해소)
- [x] `MicroBatchSpec.kv_rows_total` 필드 신설 + `admission.layer1` 의 산출
- [x] `dispatcher` 의 `micro_batches` dict + `register`/`unregister` API (Q1-bis) + `_op_time` PIM branch 의 실 signal flow (placeholder default args 제거)
- [x] `main_loop.ADMISSION_TICK` body 의 `MicroBatchSpec` → `MicroBatch` 변환 (Q1) + `dispatcher.register` 호출

**Unit Tests:** (신규 ~167 + 기존 315 regression-fixed = **482 passed**)
- [x] `Instance` — GPU pool · PIM pool 자원 점유 / 해제 round-trip (`test_instance.py` 11 passed)
- [x] `InstancePipeline` — A → B → A_next dispatch 순서 + steady_state max(A,B) + init guard (Case A 16 GPU) (`test_instance_pipeline.py` 35 passed)
- [x] `NVLinkTransfer.time` — decode + prefill shape 산식 정합 + coef linearity + monotonicity + determinism (`test_nvlink.py` 18 passed)
- [x] Async transfer hiding — `test_nvlink_async_hidden_invariant` (Q7 dummy 영역 prefigure, visible NVLink sensitivity 는 Impl-10)
- [x] Steady-state cycle — A_cycle · B_cycle parametrize (A>B / A<B / A=B) + negative reject + 1000-call determinism
- [x] Fixed-shape gate — decode-only · prefill-uniform · mixed pass + ragged reject + cross-product parametrize (`test_instance_pipeline.py`)
- [x] `ForwardPass.run` — L ∈ {1, 8, 32, 80} parametrize (`test_forward_pass.py` 19 passed)
- [x] `LayerState` — advance 단조 + L 도달 token decode signal trigger + negative reject
- [x] **Cross-module integration** — admission → MicroBatch 변환 → dispatcher.register → dispatch_pim 의 실 chain (`test_cross_module_pipeline.py` 19 passed)
- [x] **F3 source 구조 비교** — throughput ratio 8-cell sweep + A=B balance extremum (ARCH §5.7 + §6.4 정성 ground)
- [x] **Cross-module determinism** — chain 1000-iter bit-exact + seed sweep invariance (PLAN §0 C5 + ARCH §3.5.2)
- [x] **Multi-mb stress (Impl-2 패턴 정합)** — Instance resource × 50 cycle + register/unregister × 20-30 + LayerState × 20-mb + InstancePipeline stateless × 100 + NVLink stateless × 100 + composite (`test_stress_pipeline_invariants.py` 11 passed)
- [x] **Meta-test** — `_EXPECTED_MODULES` (instance · nvlink · instance_pipeline · forward_pass 추가) + Impl-5 field literal + Case A 16 GPU literal (`test_meta.py` 23 passed)
- [x] **Signature divergence lock-in** — InstancePipeline no L-loop / NVLink pure / Dispatcher.register signature / Instance.has_pim / LayerState.advance signature (`test_meta_arch_signature_divergence.py` 13 passed)

**Acceptance:** ✓ F3 정성 prefigure (throughput ratio sweep + balance extremum, ARCH §5.7 + §6.4). ✓ Instance B 입력 항상 fixed shape (ragged reject cross-product). ✓ ForwardPass 가 L-layer iteration meta-count + token decode signal trigger (L=80 default — 실 instance_pipeline.dispatch 통합은 Impl-9). ✓ Impl-4 carry-over O4.1 해소 (실 signal flow `admission → register → dispatch_pim → pim_executor.op_time`). ✓ ARCH §3.4 · §5.2 · §5.7 · §6.4 literal lock-in. ✓ Multi-mb stress invariants 보존 (Impl-2 패턴). ✓ Cross-module determinism (PLAN §0 C5). **정량 fidelity (visible NVLink sensitivity · F3 정량 ratio) 는 Impl-10 / Phase 3 영역 deferred (§0.5 정합).**

---

### Impl-6 — Trace Replayer + Completion Handler ✓ (commit pending)

> **보유 long-ctx production trace 시범 ingest.** Impl-9 acceptance 까지는 synthetic trace 만으로 자족 가능하나, 본 phase 에서 **보유 trace 를 1 회 ingest → schema 설계 정합 + 분포 sanity check 권고**. Real workload 의 본격 sweep 사용은 Impl-10 이지만, schema reading + parsing 검증은 Impl-6 시점이 자연스러움. Vidur 작업 시 변환해 둔 원본 데이터 재활용 가능 (Vidur-specific wrapping 제거 + 우리 `TraceReplayer` schema 로 adapt).

**Implementation:**
- [x] `TraceReplayer.load(path)` — long-ctx production trace ingest (CSV schema, Q2 — longbench_csv only)
- [x] `TraceReplayer.replay(rate_multiplier)` — arrival time scaling + Request generator (max_tokens = num_decode_tokens, Q6 hybrid)
- [x] KV length 분포 · arrival rate 통계 산출 (`stats()` — TraceStats dataclass)
- [x] 1M-class benchmark dataset adapter (long-doc) — Phase 3 영역 (NotImplementedError stub, Q8)
- [x] Mid-ctx production chat trace adapter — Phase 3 영역 (NotImplementedError stub, Q8)
- [x] `Completion.check(req, eos_seen=False)` — EOS marker (Q6 hybrid) 또는 max_tokens 도달 검사 (idempotent)
- [x] `Completion.finalize(request)` — KV slot 회수 (`kv_accountant.release`, Q7 직후) + completion_time 기록 + state → COMPLETED
- [x] `Request` lifecycle field 신설 — `max_tokens` · `decoded_count` · `completion_time` (Q10 (b) Request = lifecycle owner)
- [x] `SchedulerCore._maybe_advance_forward_pass` — KERNEL_COMPLETION(O_PROJ) → LayerState.advance → L 도달 시 token decode signal consumer (Q5 — main_loop 영역)
- [x] `implementation/data/longctx_longbench_lambda_{3_40, 6_67}.csv` 신설 — 외부 Vidur 변환 trace 의 self-contained copy (Q1)

**Unit Tests:** (총 170 신규 passed; 기존 482 + 신규 170 = **652** — `implementation/plans/impl_6.md` 참조)
- [x] `TraceReplayer.load` — fixture trace 파일 schema 검증 + parsing round-trip + 5 malformed cross-product fail-fast + Phase 3 stub raise (Q8) + 실 trace (12,279 + 24,054 row) ingest 정합
- [x] `TraceReplayer.replay` — arrival time scaling (rate sweep + extreme value R10) + max_tokens / kv_length / prompt_tokens 정합 + R6 generator one-shot semantic
- [x] KV length 분포 — load → replay 자기 일치 (D=0 자명) + R1 실 trace stats 값 hardcoded lock-in (regression detection)
- [x] Arrival rate 통계 — stats() 의 mean / std 산식 정합 + R1 실 trace 값 lock-in
- [x] Determinism — 동일 path 1000-iter load bit-exact + seed independence (RNG 의존 0)
- [x] `Completion.check` — max_tokens boundary (above · equal · below · zero) + EOS branch (Q6 hybrid) + 5×3×2 cross-product 전수 + idempotent (COMPLETED True) + pure (no mutation)
- [x] `Completion.finalize` — KV release (Q7) + completion_time (clock.now) + state → COMPLETED + atomic (release 실패 시 state 보존) + double finalize raise + PENDING raise + 50-roundtrip 누수 0
- [x] **Cross-module integration** (R3·R4·R7 보강) — trace → admission → register → dispatch → completion → release 진정한 chain + 50/100-req lifecycle KV no-leak + 실 trace capacity boundary (default reject + bumped admits) + finalized req mb 잔존의 correctness invariant
- [x] **Multi-mb stress** (R8 보강) — (I-F1)~(I-F6) 6 invariant 위 100-roundtrip + 50-mb decoded_count signal + completion_time single-set + in_flight_requests no orphan + Q10 mb.decode_tokens 불변 + composite seed sweep (4 cell)
- [x] **Meta-test + signature divergence lock-in** — Q1~Q10 결정 영구 기록 (TraceReplayer RNG 부재 · Completion dispatcher 부재 · Request lifecycle fields · SchedulerCore in_flight_requests · supported/Phase-3 schemas inventory · data directory presence · ARCH §3.3 KV resident invariant)

**Acceptance:** ✓ Replay 출력의 KV length + arrival rate 통계 정합 (R1 lock-in). ✓ Completion 후 KV slot 누수 0 (50/100 req lifecycle · 실 trace 100 req). ✓ Token decode signal chain 정합 (R2 EOS path + R5 multi-token step-by-step). ✓ Impl-5 carry-over O5.6 해소 (Request = lifecycle owner, MicroBatch.decode_tokens 는 dispatch metadata placeholder 유지). ✓ ARCH §3.3 · §8 literal lock-in. ✓ Q1~Q10 결정 정합 lock-in (9 signature meta-tests). ✓ PLAN §0 C1·C3·C5 prefigure (50-req all-completed + KV no-leak + deterministic seed). ✓ 모든 unit test green (652 passed, regression 0). **정량 fidelity (외부 reference distribution 과의 KS test) 는 Impl-10 deferred (§0.5 정합, §7 O6.9)**.

---

### Impl-7 — (Removed)

**Status:** Removed from plan scope. 사유: §1.2 Out of Scope — comparative baseline scheduler reimplementation (Sarathi-Serve · vLLM) 본 RFC deliverable 외 (Deliverables 정합).

본 slot 은 Impl-8 / Impl-9 / Impl-10 번호 추적성 유지 위해 stub 으로 유지 — 번호 재배치 안 함.

---

### Impl-8 — Structural Evaluator (Dispatch Trace + Convergence + F1~F5 Decomposition) ✓ (commit pending)

> **Deliverables 정합.** 본 phase 는 D1 (Impl-9 시뮬레이터 통과) 의 *증거 산출* + D2 (Impl-10 calibrated projection) 의 *schema 골격*. 절대 metric (TTFT · TPOT · throughput · goodput) 은 §1.2 Out of Scope.

**Implementation:**
- [x] `Evaluator.dispatch_trace` — §6.5 Init/T1~T5 sequence 캡처 (DispatchEvent dataclass: timestamp · mb_id · node_type · resource · k_total · dag_state_snapshot)
- [x] `Evaluator.admission_convergence` — deadband 위 idle fraction 시간 series + ConvergenceVerdict (converged · oscillating · in_band_fraction · samples)
- [x] `Evaluator.idle_fraction` — Instance A scope (GPU · PIM 2 자원). Per-instance A/B split 은 Impl-9 (O8.1 carry-over)
- [x] `Evaluator.pim_utilization` — `Σ k_total · dt / (k_max · total_time)` aggregate channel-time
- [x] `Evaluator.pipeline_efficiency` — `max(A, B) / (A + B)` ratio (a/b ≤ 0 reject)
- [x] `Evaluator.acceleration_decomposition` — F1·F2·F3·F5 cycle ratio direction 표 (D2 schema 골격, F4 미포함 = ARCH §5.7 precondition)
- [x] `Evaluator.report` — Python dict (8 key) + markdown 표 (PULS 단독, Comparative baseline 없음)
- [x] D1 hook API — `Dispatcher.on_dispatch` + `SchedulerCore.on_admission_tick` (Evaluator standalone 정합, D3)
- [x] F1~F5 ablation flag wiring — `config.AblationConfig` (D2 정합) + dispatcher F1 분기 + window F2 capacity override + evaluator F3 직접 산식 + pim_emulator F5 분기 (kv_rows_lockstep = max_kv × num_decode_reqs)

**Unit Tests:** (총 ~110 신규 passed; 기존 652 + 신규 ~110 = ~760+ — `implementation/plans/impl_8.md` 참조)
- [x] `Evaluator.dispatch_trace` — §6.5 P/M/N fixture 위 Init/T1~T5 sequence 정확 재현 (cluster B 12 tests)
- [x] `Evaluator.admission_convergence` — 합성 oscillation / 수렴 / boundary trace 판정 정확 (cluster C 10 tests)
- [x] `Evaluator.idle_fraction` — Instance A scope (gpu/pim 2 자원) bit-exact telemetry (cluster A)
- [x] `Evaluator.pim_utilization` — Σ k·dt 산식 정합 + [0,1] boundary (cluster A + F)
- [x] `Evaluator.pipeline_efficiency` — max/(a+b) 산식 + balance/extremum boundary (cluster A)
- [x] `Evaluator.acceleration_decomposition` — F1·F2·F3·F5 산식 정합 + direction_positive + F4 미포함 lock-in (cluster A + D 17 tests)
- [x] Reproducibility — 동일 seed + 동일 trace → bit-exact report (cluster A + E + F multi-seed sweep)
- [x] **D1 hook chain** — dispatcher.on_dispatch + scheduler_core.on_admission_tick → record_dispatch/record_admission_tick chain (cluster E 10 tests)
- [x] **Non-intrusion** — Evaluator 부착 전후 KV invariant + DAG state bit-exact (cluster E + F)
- [x] **Multi-mb stress** — 100-mb dispatch + 100 admission tick 위 capture loss 0 + I-E1~I-E5 invariant 보존 (cluster F 19 tests)
- [x] **Comparative baseline 부재 lock-in** — Evaluator 의 method · field 에 baseline/sarathi/vllm/compare + ttft/tpot/throughput/goodput substring 부재 meta-test (cluster G + H 4 tests)
- [x] **F4 precondition lock-in** — AblationSource enum == {F1, F2, F3, F5}, decomp cell 4 만 (ARCH §5.7 literal, cluster D + G)

**Acceptance:** ✓ PULS scheduler 의 구조 산출 (dispatch trace · 수렴 trace · F1~F5 decomposition) schema 정합 + reproducibility 성립 (PLAN §0 C5 prefigure). ✓ §6.5 Init/T1~T5 fixture reproduction (12 tests). ✓ §6.4 deadband convergence heuristic 정합 (10 tests). ✓ F1·F2·F3·F5 ablation direction + 산식 정합 (17 tests). ✓ *Comparative baseline 미산출 lock-in* — Evaluator method/field 부재 meta-test 4. ✓ *절대 metric 미산출 lock-in* — TTFT/TPOT/throughput/goodput substring 부재. ✓ Multi-mb stress (100-mb × 4 seed) 위 I-E1~I-E5 invariant 보존. ✓ ARCH §5.7 · §6.4 · §6.5 · §6.7 literal lock-in. **정량 ratio 절대값 산출 0 (Impl-10 deferred — PLAN §0.5).**

---

### Impl-9 — End-to-end Driver (Completeness Checkpoint)

**본 phase 의 acceptance 통과 = §0 정의에 의한 scheduler 완성 시점.**

**Implementation:**
- [ ] `Run.init(config_path)` — config load → 모든 모듈 instantiate → trace open → initial state setup
- [ ] `Run.step()` — outer iteration: admission tick → dispatch → event drain → completion handling. Per-step 단위는 admission cadence 와 일치
- [ ] `Run.loop()` — termination condition (`RequestQueue` empty AND `InFlightWindow` empty) 까지 step 반복
- [ ] `Run.teardown()` — metric report → cleanup
- [ ] Entry point (binary `puls-sched` 또는 module main) — config path · trace path · output path 인자 받음

**Unit Tests:**
- [ ] `Run.init` — 모든 module 정상 instantiate; config 부정확 시 fail-fast
- [ ] `Run.step` — 1 step 후 적어도 1 event progress (clock advance 또는 node state transition)
- [ ] `Run.loop` — bounded synthetic 10-request trace 위 finite 시간 내 termination
- [ ] `Run.teardown` — metric report 정합 출력 (schema validation)
- [ ] Determinism — 동일 config + 동일 trace + 동일 seed → 두 번 run 시 bit-exact output

**Acceptance — Completeness Acceptance Test (§0):**

- [ ] **C1.** Synthetic 100-request trace → 모든 request 가 EOS 또는 max_tokens 도달로 종료
- [ ] **C2.** Schema-valid 한 *구조적 산출* — §6.5 dispatch trace (Init/T1–T5) 재현 + adaptive admission deadband 수렴 trace + F1·F2·F3·F5 ablation cycle ratio 표. *Comparative baseline 미산출 (Deliverables 정합).*
- [ ] **C3.** KV slot 누수 없음 — completion 후 capacity 회수 정상 (`kv_accountant.remaining` 이 trace 종료 시 initial capacity 와 일치)
- [ ] **C4.** Invariant (I1–I5) 위반 0 회 over full trace
- [ ] **C5.** Determinism — 동일 seed + trace → bit-exact 구조 산출

**이 phase 의 5 acceptance 동시 충족 = scheduler runnable.** 이후 추가 발견되는 gap 은 plan 갱신 영역.

---

### Impl-10 — F1~F5 Calibrated Projection (Phase 3 영역, D2 산출)

> **Calibrated Projection Phase.** Impl-1~9 의 dummy time model 을 calibrated input 으로 교체 → F1·F2·F3·F5 각 source 의 가속 ratio 를 workload regime 격자 위에서 산출 (D2 deliverable). *Silicon-validated PULS measurement 아님* — PULS 실리콘 부재로 PIM side 는 Ramulator2 추정 (JEDEC spec scaling, `ramulator2_hbm4_estimated_jedec_spec` 라벨 동반) 유지. GPU side 만 lab 블랙웰 8 GPU 실측. 코드 영향 없음 (`gpu_executor` · `pim_executor` lookup table 만 교체). *Comparative baseline 산출 없음 — F1~F5 자체의 ratio 가 deliverable (Deliverables 정합).*

**Calibration source:**

- GPU side (`t_proj` · `t_FFN` · `t_attn_GPU`): lab 블랙웰 8 GPU 실측
- PIM side (`t_PIM` · SP-PIM aggregate · broadcast overhead): Ramulator2 추정 ingest (출처 라벨)
- NVLink: 블랙웰 SXM 실측 또는 spec 인용

**Implementation:**
- [ ] Workload sweep — ctx ∈ {2k, 8k, 32k, 128k, 512k, 1M} × batch ∈ {16, 64, 128, 256}
- [ ] k_total sweep — fixed k_total 대조군 + adaptive k_total 비교
- [ ] Chunk size sweep — PULS 내부 admission 의 chunk 결정 sensitivity (외부 reference 없음)
- [ ] Deadband width sweep — ctx-tiered lookup vs static 비교
- [ ] F1·F2·F3·F5 가속 source 별 ablation 기여도 분해 + F4 (steady-state 전제) 충족 검증
- [ ] D2 산출 — F1~F5 cycle ratio 표 (workload regime 격자 cell 별)

**Unit Tests:**
- [ ] Sweep grid coverage — ctx × batch 격자 모든 셀 실행 확인 (누락 0)
- [ ] F1 ablation — SP-PIM 비활성화 (GPU attention kernel route) 시 가속 source disappear
- [ ] F2 ablation — Double-buffering 비활성화 (μ-batch 직렬 강제) 시 `A_cycle = t_proj + t_attn`
- [ ] F3 ablation — Single-instance fallback (A·B fusion) 시 steady-state cycle = `A_cycle + B_cycle`
- [ ] F5 ablation — Channel-independent scheduling 비활성화 (lock-step max-KV wait) 시 straggler bubble 복원
- [ ] F4 검증 — F2·F3 활성화 + μ-batch staggering 활성화 시 steady-state regime 도달 (F4 는 별도 기여가 아닌 전제 충족 확인)
- [ ] 출처 라벨 round-trip — calibrated input 의 `source` 필드 (`lab_blackwell_measured` · `ramulator2_hbm4_estimated_jedec_spec` · `nvlink4_sxm_spec`) 가 D2 보고서까지 보존

**Acceptance:** §5.7 F1·F2·F3·F5 각 source 의 isolated cycle ratio 가 calibrated input 위 workload regime 격자에서 산출. F4 steady-state 전제 충족. *Comparative baseline 미산출 — D2 deliverable 단독 (Deliverables 정합).*

---

## 5. Validation Strategy

| 검증 layer | 항목 | 방법 |
|---|---|---|
| **Unit** | Module-level invariant 보존 | 각 Impl-N 의 *Unit Tests* 체크리스트 (§4) |
| **Integration** | DAG topological order | Cycle detection + ordering verifier |
| **Integration** | §6.5 dispatch trace 재현 | Init / T1–T5 fixture trace 재현 test |
| **Integration** | Invariant 위반 0 회 | Synthetic 100 μ-batch stress trace + assertion 강제 |
| **E2E** | **End-to-end runnable** | **§0 Completeness Acceptance Test (C1–C5) — Impl-9 acceptance 와 일치** |
| **Calibration** | Time model fidelity | **Impl-10 / Phase 3 영역. Impl-1~9 의 validation scope 에서 제외 (§0.5 Numeric Value Policy 정합) — Phase 0 추정값과의 정량 일치 검증 금지** |
| **Reproducibility** | 구조 산출 determinism | 동일 seed + 동일 trace + 동일 scheduler → bit-exact 구조 산출 (dispatch trace · 수렴 trace · F1~F5 ratio) |

각 layer 는 독립 — unit test 통과가 integration 보장 아니며, integration 통과가 E2E runnable 보장 아님. CI 영역에선 unit · integration · E2E 자동화; calibration · reference 는 manual gate.

---

## 6. Open Issues

- **OI1. SP-PIM cross-GPU cooperation 시간 model.** Ramulator2 single-stack scope → 2048-channel lock-step timing 추가 모델링 필요. Impl-4 의 broadcast overhead 항목.
- **OI2. ~~GPU op time 의 잠정/확정 분리~~ → §0.5 Numeric Value Policy 로 흡수.** Impl-1~9 는 dummy placeholder 만 사용 (spec-based 잠정값 포함 금지). 실측 / 추정값 주입은 Impl-10 단일 시점. 예외 = RTL FSM cycle 단독 (회로 합성 확정값, config 하드코딩 OK).
- **OI3. Comparative baseline reimplementation scope 제외.** Pre-HW dummy time 위 reproduction 정합 판단 불가 + Phase 3 calibrated projection 후에도 PULS silicon 부재로 양방 실측 비교 불가능. RFC deliverable 을 *D1 (동작하는 scheduler) + D2 (F1~F5 source decomposition)* 으로 정의 (Deliverables · §1.2 정합). Comparative axis 미포함.
- **OI4. Deadband σ 측정 불가.** Self-authored framework 에 hardware jitter model 부재 → balanced regime 정성 거동만 측정 (§6.4 disclosure 정합).
- **OI5. 구현 언어 미결.** Python 권장 (vLLM · Sarathi-Serve 정합) 이나 simulation throughput 영역에서 Rust / Go 대안 검토 가능. Impl-1 진입 시 결정.
- **OI6. 추가 module 발견 가능성.** §0 의 iterative discovery 원칙 — Impl-9 E2E acceptance 실행 시 추가 누락 노출 가능. 본 plan 은 *시작점* 이지 final spec 아님. Gap 노출 시 plan 갱신으로 흡수.
- **OI7. Phase 3 의 silicon validation 부재 disclosure.** Impl-10 후에도 PULS 자체 실리콘 부재로 PIM side (`t_PIM` · SP-PIM aggregate · broadcast overhead) 는 Ramulator2 추정 유지 (`ramulator2_hbm4_estimated_jedec_spec` 라벨). D2 산출은 *calibrated projection* 이지 silicon-validated measurement 아님. README/ARCHITECTURE 의 "will be measured in Phase 3" 류 문구는 "will be projected with stated provenance" 로 정정 필요 (별도 follow-up commit 영역; PLAN.md 자체에는 framing 반영 완료).

---

## 7. Cross-references

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §1–§8 — 본 구현이 따르는 architecture spec
- [`../README.md`](../README.md) Phase 1 — public-facing scope
