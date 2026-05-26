# PULS Scheduler — Implementation Plan

**Working draft (2026-05-26).** References: [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §5–§6, [`../README.md`](../README.md) Phase 1.

본 문서는 README Phase 1 / ARCHITECTURE §5.5 · §6.7 의 self-authored scheduler framework 의 구현 계획 세부서. Module 분해 + phase 별 *Implementation* (구현 산출) + *Unit Tests* (단위별 검증) + *Acceptance* (phase 진입 조건) 의 항목화.

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
- **C2.** Non-NaN · finite 한 TTFT · TPOT · throughput metric 출력
- **C3.** KV slot 누수 없음 — completion 후 capacity 회수 정상
- **C4.** Invariant (I1–I5) 위반 0 회 over full trace
- **C5.** Determinism — 동일 seed + trace → bit-exact metric

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
- **Impl-7** — Baseline scheduler. PIM path 비활성화 grep meta-test + 동일 framework reproducibility (PULS · baseline 양자)
- **Impl-8** — Evaluator. Metric 정의 round-trip + SLO boundary parametrize (TTFT · TPOT) + idle fraction 정의 충돌 0
- **Impl-9** — End-to-end driver. §0 C1–C5 acceptance 각각 별도 test + full trace replay bit-exact determinism
- **Impl-10** — Sensitivity sweep. Sweep grid coverage meta-test (ctx × batch 누락 0) + ablation flag round-trip + F1·F2·F3·F5 isolation 검증

각 Impl 의 plan 문서 (`implementation/plans/impl_N.md`, local-only) 에서 위 reminder 를 출발점으로 구체 test 항목 확장. Self-review (commit 직전 mechanical 검증 — naming grep · import 순환 · LOC ceiling · plan diff · CLAUDE.md §2 sanity) 는 모든 Impl 공통.

### MODULES.md 갱신 (Impl-N 별 공통 reminder)

각 Impl 의 commit 직전 (Self-review 동시 영역) 에 `implementation/MODULES.md` 의 해당 Impl 섹션을 갱신 — 그 Impl 에서 신설 또는 변경된 모듈의 한 줄 역할 설명 추가/수정. 신설 모듈이 없는 Impl (skeleton 에 logic 만 채우는 경우) 도 기존 모듈의 한 줄 설명 갱신 필수 (예: Impl-2 의 `main_loop.py` 는 skeleton → 실 dispatcher 로 의미 변경, `dispatcher.py` 신설 — MODULES.md 에 반영).

---

## 1. Scope

### 1.1 In Scope

- μ-batch composition + token-level admission control
- Event-driven DAG dispatcher (§6.3)
- Adaptive admission with hysteresis deadband (§6.4)
- k_total decision (PIM-side dial, §5.1)
- Instance A/B inter-instance pipeline dispatch (§3.4 · §5.2)
- PIM executor emulator (Ramulator2 cycle wrapping)
- Baseline scheduler reimplementation (continuous batching · chunked prefill)
- Trace replayer + evaluator + metric collection
- End-to-end driver + request lifecycle + KV accounting (§0 완전성 충족용)

### 1.2 Out of Scope

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
| `baseline/` | vLLM-style continuous batching + Sarathi-Serve-style chunked prefill (no-PIM 비교군) | — |
| `eval` | TTFT · TPOT · throughput · goodput · idle fraction · PIM utilization | §6.7 |
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

### Impl-2 — Invariants + DAG Dispatcher

**Implementation:**
- [ ] `Invariants.check_I1` — prefill-attn(X) 가 QKV(X) 완료 후만 ready
- [ ] `Invariants.check_I2` — decode-attn(X) 가 QKV(X) 완료 후만 ready
- [ ] `Invariants.check_I3` — O-proj(X) 가 prefill-attn(X) ∧ decode-attn(X) 둘 다 완료 후만 ready
- [ ] `Invariants.check_I4` — 시점 t 에 GPU GEMM/attention op 1 개만 active
- [ ] `Invariants.check_I5` — 시점 t 에 PIM decode-attn op 1 개만 active (multi-head · multi-request batching 은 op 내부 자유)
- [ ] `Dispatcher.refresh_ready` — 모든 precedence done 인 미실행 노드 집합 산출
- [ ] `Dispatcher.pick_gpu` — priority dequeue: O-proj > prefill-attn > QKV. Tie-break: oldest μ-batch first
- [ ] `Dispatcher.pick_pim` — PIM ready set 에서 oldest μ-batch
- [ ] `Dispatcher.dispatch_gpu` / `dispatch_pim` — executor 호출 + node state → running
- [ ] Look-ahead / back-fill emergence — 명시 정책 없이 priority dequeue 의 자연 산출 (§6.5 trace 정합)

**Unit Tests:**
- [ ] I1 — QKV(X) done = False 상태에서 prefill-attn(X) dispatch 시도 시 차단
- [ ] I2 — QKV(X) done = False 상태에서 decode-attn(X) dispatch 시도 시 차단
- [ ] I3 — prefill 또는 decode 둘 중 하나만 done 일 때 O-proj 차단; 둘 다 done 일 때만 ready
- [ ] I4 — GPU 활성 op 1 개 상태에서 두 번째 GPU op dispatch 시도 시 차단
- [ ] I5 — PIM 활성 op 1 개 상태에서 두 번째 PIM decode-attn dispatch 시도 시 차단
- [ ] Priority — O-proj / prefill-attn / QKV 모두 ready 일 때 O-proj 선택
- [ ] Tie-break — 동일 priority 다중 μ-batch ready 일 때 oldest μ-batch 선택
- [ ] §6.5 trace fixture — {P, M, N} 3 μ-batch 구성 + deterministic op time 입력 → Init / T1–T5 dispatch 시퀀스 재현

**Acceptance:** I1–I5 위반 0 회 over 100 μ-batch synthetic stress trace. §6.5 dispatch trace (Init / T1–T5) 재현.

---

### Impl-3 — Admission Controller + k_total Decision + Request Queue + KV Accounting

**Implementation:**
- [ ] `RequestQueue` — waiting queue; FIFO admission; bounded queue overflow policy (reject 또는 backpressure)
- [ ] `KVAccountant` — aggregate KV slot capacity 추적; `admit(req)` 시 `kv_demand ≤ remaining` 검증; completion 시 `release(req)` 회수
- [ ] `IdleTelemetry` — GPU · PIM idle fraction per-iteration 누적; instance A · B 별 분리 측정
- [ ] `Deadband` — ctx-tiered static lookup (short 2k–8k · mid ~32k · long 128k–1M; §6.4 표 정합)
- [ ] `Admission.layer1` — μ-batch composition 결정: prefill chunk 토큰 수, decode batch size, N. Candidate pool = `RequestQueue`. 자원 제약 = `KVAccountant`
- [ ] `Admission.mfu_floor` — `N ≥ N_sat` (FFN GEMM saturating knee) 강제
- [ ] `Admission.balance_inter_AB` — `A_cycle` vs `B_cycle` 차이 기반 prefill chunk admit 조정
- [ ] `Admission.balance_intra_A` — GPU vs PIM idle fraction 기반 decode / prefill chunk admit 조정
- [ ] `kTotalDecider.solve(t_proj, t_PIM_fn, N_decode)` — `max k_total s.t. t_PIM(k_total, N_decode) ≤ t_proj`; **k_total ∈ {0, 256, 512, ..., 2048}** (aggregate, 8 GPU lock-step 전제, 9-step dial); per-GPU `n × 32, n ∈ {0..8}` → aggregate `k_total = 8 × n × 32`
- [ ] `kTotalDecider.over_budget_handler` — `t_PIM(2048) > t_proj` 진입 시 admission layer 1 escalation (decode 축소 또는 prefill chunk 추가)

**Unit Tests:**
- [ ] `RequestQueue` — FIFO 동작; bounded queue overflow handling (overflow 시 reject 또는 backpressure signal)
- [ ] `KVAccountant` — capacity 초과 admit 시 reject; completion 후 `release` 호출 시 capacity 정상 회수
- [ ] `IdleTelemetry` — 합성 event sequence (활성 / idle 구간 명시) 위에서 idle fraction 계산값 정확
- [ ] `Deadband` — 각 ctx tier (short / mid / long) lookup 의 deterministic 출력
- [ ] `Admission.layer1` — 동일 input → 동일 N · chunk size 출력 (reproducibility)
- [ ] `Admission.mfu_floor` — `N < N_sat` 입력 시 N_sat 으로 clamp 또는 reject
- [ ] `Admission.balance_inter_AB` — A_cycle > B_cycle 시 prefill chunk admit 증가 방향
- [ ] `Admission.balance_intra_A` — GPU idle > θ_high 시 decode admit 증가; PIM idle > θ_high 시 prefill chunk admit 증가
- [ ] `kTotalDecider.solve` — 9-step k_total 영역에서 max k 선택 정확성; 동일 입력 → 동일 출력
- [ ] `kTotalDecider.over_budget_handler` — `t_PIM(2048) > t_proj` 시 escalation signal 발사 검증
- [ ] Stack-granularity 검증 — `k_total mod 256 = 0` (per-GPU n × 32 with same n across 8 GPUs)

**Acceptance:** Balanced regime 에서 idle fraction 이 deadband 내 oscillation 없이 수렴. `kTotalDecider` 출력이 결정론적. KV capacity overflow / underflow 발생 0 회.

---

### Impl-4 — PIM Executor Emulator

**Implementation:**
- [ ] `PIMExecutor.tile_time(regime)` — FP8 (compute-bound) / FP16 (load-bound) regime 분기 → tile-level cycle 산출
- [ ] `PIMExecutor.op_time(k_channels, N_decode, N_kv_avg)` — SP-PIM aggregate (Q-replicate + KV-row sharding) → op-level time
- [ ] `PIMExecutor.dispatch(node)` — blocking event 발사; completion timestamp = current + op_time
- [ ] Ramulator2 cycle 데이터 loader (JSON / CSV ingest) — **loader 로직만 구현. 실제 Phase 0 Ramulator2 HBM4 추정 cycle 의 ingest 는 Impl-10 영역. Impl-4 단계의 dev/test 는 fixture dummy 데이터 사용 (§0.5 Numeric Value Policy 정합)**
- [ ] SP-PIM cross-GPU lock-step cooperation overhead model (8 GPU broadcast latency) — **로직만, latency 값은 config placeholder dummy**
- [ ] FSM determinism 보장 — jitter ±0, 동일 입력 → 동일 cycle count

**Unit Tests:**
- [ ] `PIMExecutor.tile_time` — FP8 vs FP16 regime 의 cycle ratio ≈ 2 (§3.1, §6.6 정합)
- [ ] `PIMExecutor.op_time` — k_channels sweep 에서 monotonic 감소 (KV-row sharding 정합)
- [ ] `PIMExecutor.op_time` — N_kv_avg sweep 에서 monotonic 증가 (KV 길이 비례)
- [ ] `PIMExecutor.dispatch` — completion timestamp = current_time + op_time (deterministic)
- [ ] Ramulator2 loader — fixture JSON / CSV 위에서 schema 검증 + cycle field round-trip
- [ ] SP-PIM broadcast overhead — k_total = 2048 (cross-GPU) vs k_per_gpu = 256 (single-GPU only) 시간 차이가 broadcast latency 만큼
- [ ] FSM determinism — 동일 입력 1000 회 호출 → bit-exact cycle count (jitter ±0)

**Acceptance:** FP8 / FP16 regime 분기 로직이 결정론적 (동일 입력 → 동일 regime 선택). Tile time 출력이 config `pim_tile_time` placeholder 와 정확 일치 (lookup 정합). FSM 결정론 — 동일 입력 1000 회 호출 → bit-exact cycle count. **정량 fidelity (Ramulator2 추정값과의 정합) 는 Impl-10 / Phase 3 영역으로 deferred (§0.5 Numeric Value Policy 정합).**

---

### Impl-5 — Instance A/B Pipeline + Forward Pass + Inter-instance Handoff

**Implementation:**
- [ ] `Instance` 클래스 — GPU pool + (Instance A 한정) PIM pool
- [ ] `InstancePipeline` — A → B → A_next 순서 관리; 양방향 NVLink handoff
- [ ] `NVLinkTransfer.time(tensor_shape)` — `config.nvlink_time_per_byte × bytes(tensor_shape)` 계산 로직. decode shape `[B × hidden]` + uniform-chunk prefill shape `[(B · chunk) × hidden]` 양 case 지원. **계수 자체는 config placeholder (Impl-10 에서 실측 / spec 인용으로 교체). §5.2 정합.**
- [ ] Async transfer hiding — A_cycle / B_cycle 내 transfer time 흡수
- [ ] Steady-state pipeline cycle measurement = `max(A_cycle, B_cycle)`
- [ ] Fixed-shape handoff 강제 — Instance B 가 ragged batching 미수신 (§5.2)
- [ ] `ForwardPass.run(μ_batch)` — L-layer iteration loop: `for layer_idx in range(L): instance_pipeline.dispatch(μ_batch, layer_idx)`
- [ ] `LayerState` — μ-batch 의 current_layer_index 추적; L 도달 시 token decode signal 발사

**Unit Tests:**
- [ ] `Instance` — GPU pool · PIM pool 자원 점유 / 해제 round-trip
- [ ] `InstancePipeline` — A → B → A_next dispatch 순서 검증 (layer index 단조 증가)
- [ ] `NVLinkTransfer.time` — decode shape `[B × hidden]` + prefill shape `[(B · chunk) × hidden]` 두 case 의 시간 산출 정확
- [ ] Async transfer hiding — `t_handoff < A_cycle` 영역에서 effective cycle = A_cycle (transfer hidden)
- [ ] Steady-state cycle — A_cycle · B_cycle 다른 값 입력 위에서 `max(A_cycle, B_cycle)` 출력
- [ ] Fixed-shape gate — Instance B 가 ragged tensor 수신 시 assertion fail
- [ ] `ForwardPass.run` — L = 32 입력 시 32 회 instance_pipeline.dispatch 호출
- [ ] `LayerState` — layer index 단조 증가; L 도달 시 token decode signal 발사

**Acceptance:** Single-instance baseline vs A-B split steady-state cycle 비교에서 F3 (inter-instance pipeline) 효과 정성 확인. Instance B 입력 텐서 항상 fixed shape. ForwardPass 가 L-layer 완주.

---

### Impl-6 — Trace Replayer + Completion Handler

> **보유 long-ctx production trace 시범 ingest.** Impl-9 acceptance 까지는 synthetic trace 만으로 자족 가능하나, 본 phase 에서 **보유 trace 를 1 회 ingest → schema 설계 정합 + 분포 sanity check 권고**. Real workload 의 본격 sweep 사용은 Impl-10 이지만, schema reading + parsing 검증은 Impl-6 시점이 자연스러움. Vidur 작업 시 변환해 둔 원본 데이터 재활용 가능 (Vidur-specific wrapping 제거 + 우리 `TraceReplayer` schema 로 adapt).

**Implementation:**
- [ ] `TraceReplayer.load(path)` — long-ctx production trace ingest
- [ ] `TraceReplayer.replay(rate_multiplier)` — arrival time scaling
- [ ] KV length 분포 · arrival rate 통계 산출 (sanity check)
- [ ] 1M-class benchmark dataset adapter (long-doc) — Phase 3 영역
- [ ] Mid-ctx production chat trace adapter — Phase 3 영역
- [ ] `Completion.check(request, decoded_token)` — EOS 토큰 검출 또는 max_tokens 도달 검사
- [ ] `Completion.finalize(request)` — KV slot 회수 (`kv_accountant.release`) + completion timestamp 기록 + state → completed

**Unit Tests:**
- [ ] `TraceReplayer.load` — fixture trace 파일 schema 검증 + parsing round-trip
- [ ] `TraceReplayer.replay` — arrival time scaling (rate_multiplier × 2 → arrival interval / 2)
- [ ] KV length 분포 — 원 trace 와 KS test p > 0.05
- [ ] Arrival rate 통계 — 원 trace 와 평균 · 분산 ±5% 이내
- [ ] Determinism — 동일 seed → 동일 replay 시퀀스
- [ ] `Completion.check` — EOS 토큰 입력 시 True; max_tokens 도달 시 True; 그 외 False
- [ ] `Completion.finalize` — 회수 후 `kv_accountant.remaining` 증가; completion_time 기록 정확

**Acceptance:** Replay 출력의 KV length 분포 + arrival rate 통계가 원 trace 와 일치. Completion 후 KV slot 누수 0.

---

### Impl-7 — Baseline Scheduler Reimplementation

**Implementation:**
- [ ] `baseline/continuous_batching` — vLLM-style: iteration-level, prefill-priority, no chunked prefill, no PIM
- [ ] `baseline/chunked_prefill` — Sarathi-Serve-style: mixed batch primitive, prefill chunk + decode 공존, no PIM
- [ ] 두 baseline 모두 동일 framework + 동일 time model 위에서 동작
- [ ] PIM dispatch path 비활성화 (decode-attn → GPU attention kernel route)

**Unit Tests:**
- [ ] `continuous_batching` — prefill ready 일 때 decode 후순위 동작 검증
- [ ] `continuous_batching` — chunked prefill 미적용 (full prefill block-and-execute)
- [ ] `chunked_prefill` — chunk size 결정 + mixed batch 구성 (prefill chunk + decode 공존)
- [ ] PIM dispatch path 비활성화 — decode-attn 가 GPU attention kernel 경로 사용
- [ ] 공통 framework reproducibility — 동일 trace · 동일 time model · 동일 seed → bit-exact metric

**Acceptance:** 원 구현 (vLLM · Sarathi-Serve) 의 published metric (TTFT · TPOT · throughput) 과 정성 일치.

---

### Impl-8 — Evaluator + Metric Collection

**Implementation:**
- [ ] `Evaluator.ttft` — per-request TTFT 산출
- [ ] `Evaluator.tpot` — per-token TPOT (decode 영역)
- [ ] `Evaluator.throughput` — tokens/s · requests/s
- [ ] `Evaluator.goodput` — SLO-attainment-weighted throughput (TTFT · TPOT SLO 동시 만족 비율)
- [ ] `Evaluator.idle_fraction` — Instance A · B 별, GPU · PIM 별
- [ ] `Evaluator.pim_utilization` — aggregate channel-time utilization
- [ ] `Evaluator.pipeline_efficiency` — `max(A, B) / (A + B)` ratio
- [ ] `Evaluator.report` — 표 + 분포 출력 (PULS vs baseline 동일 형식)

**Unit Tests:**
- [ ] `Evaluator.ttft` — synthetic request 위에서 TTFT = (first token time − arrival time) 정확
- [ ] `Evaluator.tpot` — decode token 간 평균 간격 정확
- [ ] `Evaluator.throughput` — total tokens / total elapsed 정확
- [ ] `Evaluator.goodput` — SLO 만족 (TTFT · TPOT 둘 다 SLO 안) 비율 정확
- [ ] `Evaluator.idle_fraction` — Instance / executor 별 idle fraction 정확
- [ ] `Evaluator.pim_utilization` — `Σ k_total · dt / (k_max · total_time)` 정확
- [ ] `Evaluator.pipeline_efficiency` — A_cycle · B_cycle 입력에서 `max / (A + B)` 정확
- [ ] Reproducibility — 동일 seed + 동일 trace + 동일 scheduler → bit-exact metric

**Acceptance:** PULS / baseline schedulers 의 metric 출력이 동일 schema; reproducibility 성립.

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
- [ ] **C2.** Non-NaN · finite 한 TTFT · TPOT · throughput metric 출력
- [ ] **C3.** KV slot 누수 없음 — completion 후 capacity 회수 정상 (`kv_accountant.remaining` 이 trace 종료 시 initial capacity 와 일치)
- [ ] **C4.** Invariant (I1–I5) 위반 0 회 over full trace
- [ ] **C5.** Determinism — 동일 seed + trace → bit-exact metric

**이 phase 의 5 acceptance 동시 충족 = scheduler runnable.** 이후 추가 발견되는 gap 은 plan 갱신 영역.

---

### Impl-10 — Sensitivity Sweep (Phase 3 영역)

> **Hardware 필요 시점.** Impl-1~Impl-9 은 spec-based 잠정 GPU op time 으로 동작 가능 (hardware 0 개). 본 phase 부터 **랩실 hardware (블랙웰 8 GPU) 실측 `t_proj · t_FFN · t_attn_GPU` 값으로 교체 필요** — 가속 배수 · latency 절대값 의 최종 claim 근거. 코드 영향 없음 (`gpu_executor` lookup table 만 교체).

**Implementation:**
- [ ] Workload sweep — ctx ∈ {2k, 8k, 32k, 128k, 512k, 1M} × batch ∈ {16, 64, 128, 256}
- [ ] k_total sweep — fixed k_total 대조군 + adaptive k_total 비교
- [ ] Chunk size sweep — Sarathi-Serve 권고치 ± 영역
- [ ] Deadband width sweep — ctx-tiered lookup vs static 비교
- [ ] F1·F2·F3·F5 가속 source 별 ablation 기여도 분해 + F4 (steady-state 전제) 충족 검증

**Unit Tests:**
- [ ] Sweep grid coverage — ctx × batch 격자 모든 셀 실행 확인 (누락 0)
- [ ] F1 ablation — SP-PIM 비활성화 (GPU attention kernel route) 시 가속 source disappear
- [ ] F2 ablation — Double-buffering 비활성화 (μ-batch 직렬 강제) 시 `A_cycle = t_proj + t_attn`
- [ ] F3 ablation — Single-instance fallback (A·B fusion) 시 steady-state cycle = `A_cycle + B_cycle`
- [ ] F5 ablation — Channel-independent scheduling 비활성화 (lock-step max-KV wait) 시 straggler bubble 복원
- [ ] F4 검증 — F2·F3 활성화 + μ-batch staggering 활성화 시 steady-state regime 도달 (F4 는 별도 기여가 아닌 전제 충족 확인)

**Acceptance:** §5.7 가속 source 표 의 F1·F2·F3·F5 각 source 의 isolated 측정값 정성 정합. F4 steady-state 전제 충족.

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
| **Reference** | Baseline reimpl 정확성 | 원 구현 (vLLM · Sarathi-Serve) published metric 과 정성 spot-check |
| **Reproducibility** | Metric determinism | 동일 seed + 동일 trace + 동일 scheduler → bit-exact metric |

각 layer 는 독립 — unit test 통과가 integration 보장 아니며, integration 통과가 E2E runnable 보장 아님. CI 영역에선 unit · integration · E2E 자동화; calibration · reference 는 manual gate.

---

## 6. Open Issues

- **OI1. SP-PIM cross-GPU cooperation 시간 model.** Ramulator2 single-stack scope → 2048-channel lock-step timing 추가 모델링 필요. Impl-4 의 broadcast overhead 항목.
- **OI2. ~~GPU op time 의 잠정/확정 분리~~ → §0.5 Numeric Value Policy 로 흡수.** Impl-1~9 는 dummy placeholder 만 사용 (spec-based 잠정값 포함 금지). 실측 / 추정값 주입은 Impl-10 단일 시점. 예외 = RTL FSM cycle 단독 (회로 합성 확정값, config 하드코딩 OK).
- **OI3. Baseline reimpl 의 정확성 보증.** 원 구현 코드 reading + spot-check; bit-exact reproduction 은 목표 아님 — published metric 정성 일치만 요구.
- **OI4. Deadband σ 측정 불가.** Self-authored framework 에 hardware jitter model 부재 → balanced regime 정성 거동만 측정 (§6.4 disclosure 정합).
- **OI5. 구현 언어 미결.** Python 권장 (vLLM · Sarathi-Serve 정합) 이나 simulation throughput 영역에서 Rust / Go 대안 검토 가능. Impl-1 진입 시 결정.
- **OI6. 추가 module 발견 가능성.** §0 의 iterative discovery 원칙 — Impl-9 E2E acceptance 실행 시 추가 누락 노출 가능. 본 plan 은 *시작점* 이지 final spec 아님. Gap 노출 시 plan 갱신으로 흡수.

---

## 7. Cross-references

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §1–§8 — 본 구현이 따르는 architecture spec
- [`../README.md`](../README.md) Phase 1 — public-facing scope
