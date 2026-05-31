# Phase-2 풀(pool) 모델 재설계 — 작업 플랜 (살아있는 체크리스트)

> 이 문서는 **작업하며 계속 갱신**한다. 각 항목은 `- [ ]`(미완) / `- [x]`(완료) /
> `- [~]`(진행중) / `- [!]`(막힘·재논의) 로 표시. 근거가 된 코드는 `file:line` 로 링크.
> 기준 문서: [배치_생애.md](../../배치_생애.md), [ARCHITECTURE.md](../../ARCHITECTURE.md) §5.6·§6,
> [REPORT_baseline.md](../debug_phase1/REPORT_baseline.md) §12~14.

---

## 0. 검증 — ARCH §6 정독 결과 (풀 모델 해석 확정)

- [x] **풀 모델 = ARCH 의 원래 의도임을 확인.** μ-batch 는 영속 컨테이너가 아니라
  *매 iteration 풀에서 뽑는 혼합 배치*다. 근거:
  - §6.1 (line 330): *"A μ-batch contains different requests in a phase mix … weights
    shared by all tokens; only attention branches by token-type."*
  - §6.3 (line 367): *"No explicit edges exist between distinct μ-batches — when resources
    are available, arbitrary interleaving is possible. This is the graph-theoretic basis of
    look-ahead / back fill."*
  - §6.3 (line 388): *"completion time is computed at dispatch"* → **시간 기준(computed
    wait)** 이 유휴율 반응형보다 선제·정확 (배치_생애 §밸런스와 일치).
- [x] **단, ARCH §6.4 는 *유휴율(idle-fraction)* 기반 adaptive admission 을 서술**(line 394,
  407~408). 배치_생애·STEP6 의 "시간 기준" 은 이와 모순이 아니라 *같은 수렴점의 선제
  변형*(§6.3 computed wait). → **구현은 시간 기준으로**, 유휴율은 진단 출력으로만.
- [x] **핵심 발견 (재설계 범위 대폭 축소):** DAG·dispatcher 기판은 *이미* per-iteration
  혼합 배치 + F2 backfill 을 올바로 구현. persistent-mb 병은 **`main_loop.py` +
  `admission.py` 스케줄링 레이어에만** 존재. → 기판 재사용, **한 레이어만 교체**.
- [x] **멤버십 = 디코더 분할 (확정, 사용자 결정 2026-06-01).** μ-batch 다수의 존재 이유
  = **F3 inter-AB 파이프라인**(B 가 FFN 도는 동안 A 가 다른 μ-batch attention → A 안 쉼,
  §3.4·§5.7) + F2(A 내부 proj∥attn, §5.6). 둘 다 *연속 μ-batch 가 서로 다른 요청*이어야
  성립(같은 mb 는 종속 직렬). 디코더를 한 배치에 몰면 F3·F2 둘 다 불가 → 핵심 가속원 소멸.
  ARCH §6.5 예시(P/M/N 서로 다른 요청)·window=3(2 active + 1 전이 여유)과 일치. 상세 §2.3.
- [x] **새 불변식:** 한 요청은 동시에 **최대 1개 in-flight μ-batch**에만 존재(중복
  decode-attn = KV write race + 토큰 2개). 즉 분할은 *disjoint*.

---

## 0.6 가속 요인 커버리지 — 세 가지 방식으로 *모두* 반영 (사용자 확인 2026-06-01)

> **용어:** F2 → 앞으로 **더블 버퍼링**(double buffering)으로 부른다(혼동 방지).

| 요인 | 반영 방식 | 상태 |
|---|---|---|
| **더블 버퍼링** (A 내부 GPU proj ∥ PIM attn, μ-batch 끼리) | **스케줄러 동역학** (DAG mb간 edge 0 + dispatcher) | 기판 있음 |
| **인스턴스 A∥B** (A attn ∥ B FFN, μ-batch 끼리) = F3 | **스케줄러 동역학** | **S0 에서 추가** |
| 스태거링 (위 둘의 전제) = F4 | window+DAG 자연 발현 | 있음 |
| SP-PIM (GPU attn→PIM) = F1 | **op-time 물리** (pim_emulator vs GPU fallback, ablation) | 구조 있음·fallback 수치 미보정(공시됨) |
| 채널 독립(straggler 제거) = F5 | **op-time 물리** (kv_total vs lockstep, ablation) | 있음 |
| Aux1·Aux2 (weight reuse·버스절감) | **closed-form** (evaluator) | 있음 |

→ *스케줄러 자체*가 만들어내는 동역학은 **더블 버퍼링 + 인스턴스 A∥B** 둘. 나머지는
op-time 산식·closed-form 에 이미 존재 → 잃는 것 없음. **S0 후 스케줄링 레이어가 ARCH
§3.4·§5.7 과 동역학으로 완전 정합** (F1 GPU-fallback 보정만 범위 밖, 기존 공시).

## 1. 코드 인벤토리 — 재사용(substrate) vs 교체(scheduling)

### 1a. 재사용하는 기판 (substrate — 대부분 무수정, F3 확장만 추가)

> ⚠ `node.py`·`dag.py`·`dispatcher.py`·`main_loop` 의 layer advance 는 **S0 에서 F3
> 확장**(FFN 노드·INSTANCE_B 자원, §2.6). 그 외 골격은 그대로.

- [~] `dag.py` — 4-node DAG (QKV·PREFILL_ATTN·DECODE_ATTN·O_PROJ), I1~I3 edge, mb 간
  edge 없음(자유 interleave). 풀 모델 골격 맞음. **+ S0: FFN 노드 + `O_PROJ→FFN` edge.**
- [~] `dispatcher.py` — event-driven ready-node dispatch, GPU 우선순위(O_PROJ>PREFILL>QKV),
  `_op_time` spec-derived(ns/µs 변환 정정, line 142·158), I4·I5. **+ S0: INSTANCE_B 자원.**
- [~] `node.py` — NodeType(4) / NodeState FSM. **+ S0: `NodeType.FFN`.**
- [x] `pim_emulator.py` — `op_time()` 반환 **ns**, SP-PIM tiles_per_channel 산식. 재사용.
- [x] `config.py::compute_gpu_op_time_s` / `compute_ffn_op_time_s` — per-mb spec-derived
  FLOPs/peak. PREFILL_ATTN = O(chunk×ctx) causal(line 223~232). 재사용.
- [x] `kv_accountant.py`, `completion.py`, `request.py`(lifecycle FSM), `forward_pass.py`
  (LayerState.advance), `idle_telemetry.py`, `instance_pipeline.py`, `clock/event/
  event_queue/node/request_queue/trace`, `evaluator.py`(dispatch_trace 캡처). 재사용.

### 1b. 교체·삭제 대상 (persistent-mb 스케줄링 레이어)

- [ ] `main_loop.py::_STAGGERING_TARGET_MB` (27~31) + `_per_mb_kv_budget` (495~506) —
  **재해석(삭제 아님).** (나) 분할이면 디코더를 μ-batch 들로 나누는 규칙이 *필요*. 단
  KV/2 band-aid(영속 mb 독점 방지)는 **former 의 명시적 disjoint 분할 + 활성 슬롯 목표
  (2)**로 대체. 분모 2(=동시 active μ-batch 목표)는 슬롯 수 개념으로 살아남음.
- [ ] `main_loop.py::_join_gate_open` (53) + `_try_join` (395~455) + hysteresis 게이트 —
  **삭제.** 멤버십=용량/밸런스=시간으로 대체, 합류·게이트 개념 소멸.
- [ ] `main_loop.py::_recompose_mb` (373~393) — **교체.** "같은 req 집합 재구성+합류" →
  "풀에서 per-iteration 재선택".
- [ ] `main_loop.py::_maybe_advance_forward_pass` (307~371) 의 recompose/evict 꼬리
  (360~371) — **교체.** L 도달 시 token 생성·상태전이(337~358)는 *유지*, 그 뒤 재구성
  로직만 풀 모델로.
- [ ] `main_loop.py::ADMISSION_TICK` 핸들러 (146~187) 의 "spec→영속 mb 1개 생성" —
  **교체.** per-iteration 배치 former 로.
- [ ] `admission.py::balance_intra_A` (46~62) — **삭제.** 유휴율 chunk 증량 → 시간 기준
  `balance_pim_slack` 로 일원화.
- [ ] `admission.py::balance_inter_AB` (31~44) — **보류/재검토.** inter-AB 균형 자체는
  유효하나 deadband(유휴율) 기반. 시간 기준으로 재표현할지 §2 에서 결정.
- [ ] `admission.py::layer1` (98~176) — **재작성.** head-of-line walk + KV admit 은 유지
  (멤버십=용량), per-mb KV 예산(`max_mb_kv_tokens`)·balance_intra_A 호출 제거.
- [ ] `micro_batch.py::prefill_chunk_budget` (19), `_recompose` 전제 필드 — 풀 모델에서
  per-iteration 재산출이면 영속 budget 보존 불필요. 정리 검토.

### 1c. 측정 인프라 — 미완성/공백 (Phase-2 에서 완성)

- [ ] `debug_phase1/measure_steady.py` — **스텁.** line 39 `raise SystemExit("완료 요청
  수집 경로 필요")`. 완료 요청 수집 경로 부재.
- [ ] **TTFT/TBT 캡처 substrate 부재.** `Request` 에 `first_token_time` 없음, 완료 요청을
  모으는 sink 없음(완료 시 `in_flight_requests.pop` 후 버려짐, main_loop:358). evaluator 는
  dispatch/admission trace 만 캡처(요청 단위 latency 없음).
- [ ] `debug_phase1/diag_optime.py` — line 54 `t_decode_attn_pim = ...` placeholder.
  Phase-2 판으로 정리(PIM/GPU 균형 직접 산출).

---

## 2. 설계 (pool model)

### 2.0 멤버십 = 디코더 분할 + F2 (확정 = 나)

- [x] **결정:** 풀의 decoder 를 **2개(목표) in-flight μ-batch 로 disjoint 분할**, window
  3번째 슬롯은 전이 여유. 연속 μ-batch 가 서로 다른 요청을 담아 F2 overlap (PIM 이 M
  decode-attn 하는 동안 GPU 가 M+1 QKV). ARCH §5.6·§6.5·§6.7 정합.
- [ ] **남은 세부 결정 (구현 중 확정):**
  - **재충전 방식:** 한 슬롯이 forward pass(L층) 완료 시 (ㄱ)그 슬롯 decoder 유지 +
    완료분 제거 + 풀에서 미할당 신규 충전(sticky) vs (ㄴ)풀에서 매번 재선택(refill).
    추천 **(ㄱ) sticky + 빈자리 충전** — disjoint 유지 단순, decoder 가 슬롯 간 안 튀어
    KV 안정. (ㄴ)는 완료율 편차 시 재균형 이점 있으나 disjoint 추적 복잡.
  - **TBT 정의:** = 한 decoder 가 자기 슬롯의 forward pass 한 바퀴(= 그 슬롯이 그 요청에
    연속 두 토큰 주는 간격). F2 로 슬롯들이 겹쳐 돌아 throughput↑, prefill 을 슬랙에
    숨기면 슬롯 cycle 불변 → TBT 불변.

### 2.1 자료구조 — 전역 풀

- [ ] **decode-set** = `in_flight_requests` 중 `state==DECODE` (이미 prefill 끝, 매 step
  1토큰). substrate 의 `in_flight_requests` dict 재활용.
- [ ] **prefill-queue** = `state∈{PENDING,PREFILL}` 이며 `prefill_processed <
  len(prompt_tokens)` (남은 프롬프트 chunk 필요). `request_queue`(미admit) + in_flight 의
  prefill 진행 요청.
- [ ] 한 요청은 prefill 소진 시 decode-set 으로 *전환*(빠지는 게 아님). `request.py` FSM
  (PREFILL→DECODE) 그대로.

### 2.2 per-iteration 배치 former (핵심 신규 로직)

- [ ] **멤버십 = 용량 (disjoint 분할).** 빈/완료된 μ-batch 슬롯을 재충전: 풀에서 *다른
  in-flight 슬롯에 없는* decoder 를 cap(`max_batch_size`)까지 + KV 여유만큼 prefill admit.
  활성 슬롯 목표 2(+window 3번째 전이 여유). 유휴율 게이트 없음 — "용량 있고 일감 있으면
  넣음". 한 요청 동시 1슬롯 불변식 유지.
- [ ] **밸런스 = 시간.** 이번 배치 decoder 들의 `t_pim = pim.op_time(Σkv)` 동안 GPU 슬랙
  `max(0, t_pim·margin − t_gpu_base)` 에 prefill chunk 를 끼움. base floor 없음.
  - 산식: `chunk_total = max(0, t_pim·margin − t_proj) / per_token` (현 `balance_pim_slack`
    admission.py:64~96 재사용, base floor·intra_A·inter_AB 군더더기 제거).
  - 분배: `_populate_mb_phases` 의 uniform chunk(TOTAL÷N_prefill) 유지 (ARCH §5.2).
- [ ] **prefill→decode 전이 시 풀 복귀** — 다음 iteration 부터 PIM decode 채움. 현
  `_maybe_advance_forward_pass` 의 전이 검출(345·353) 유지.
- [ ] **iteration 단위 = 1 forward pass(L=80 layer → decoder 당 1 token).** mb 의 layer
  반복(reset_micro_batch per layer)은 substrate 그대로. 바뀌는 건 L 도달 후 *재선택*.

### 2.3 staggering — μ-batch *다수의 존재 이유* = F3(일차) + F2(이차)

> **μ-batch 를 여럿 두는 *근본 이유* = F3 inter-instance 파이프라인** (사용자 강조
> 2026-06-01): 인스턴스 A(attention/proj) → B(FFN) → A(다음 layer) 가 μ-batch 1개면
> 직렬 → **B 가 FFN 도는 동안 A 가 논다.** 여러 μ-batch 면 B 가 M 의 FFN 하는 동안 A 가
> N 의 attention 을 돌려 A 를 안 쉬게 함 (ARCH §3.4·§5.7 F3, "PB1 eliminated",
> `t_A+t_B → max(t_A,t_B)`). 파이프라인 3단(A-proj / A-attn(PIM) / B-FFN) → window=3.
> F2(§5.6, A 내부 GPU proj ∥ PIM attn)는 그 위에 *추가로* 얹히는 이차 overlap.

- [ ] **F2 (intra-A):** DAG 에 mb 간 edge 없음 → PIM 이 M decode-attn 하는 동안 GPU 가
  M+1 QKV backfill (dispatcher.refresh_ready + pick_gpu). 기판 제공, 별도 코드 불필요.
- [ ] **F3 (inter-AB) — 기판 공백 ⚠.** 현재 B FFN 은 [main_loop.py:330] 의
  `instance_pipeline.dispatch(mb)` 가 `gpu_instance_b` 활동을 *기록만* 함. DAG 노드/스케줄
  종속 아님(4노드 전부 A). → "B 도는 동안 A 가 다음 μ-batch" 가 실제 스케줄에 안 걸림.
  **F3 를 측정으로 입증하려면 A→B→A 종속을 스케줄에 반영**(B-FFN 을 스테이지로 모델링,
  A 가 그 동안 다른 μ-batch attention)해야 할 수 있음 — §5 측정 설계에서 확정.
- [ ] window(capacity=3) = 2 active + 1 전이 여유. former 가 활성 슬롯을 *명시적으로*
  채워 F3·F2 가 발현하도록(현재는 admission tick 순차 충전이라 staggering 이 우연 의존).

### 2.4 admission control (동시 decoder 한도)

- [ ] decode-set 크기 > cap 이면 신규 prefill admit 정지(큐잉) — 표준 throughput/latency
  trade-off. KV 캐파 초과도 동일. 현 `layer1` head-of-line walk(109~145) 로직 재사용.
- [ ] **TTFT↔TBT 정책:** 슬랙 0(GPU 이미 PIM 보다 김)일 때 최소 prefill 보장 여부 —
  기본 *무보장*(TBT 우선, 배치_생애 §밸런스). 옵션으로 floor 노출만.

### 2.5 측정용 warm-start seed (채택 = B)

- [ ] measure 하네스에 "사전 decode 풀 seed" init 추가: t=0 에 *이미 decode 중인 요청들*
  (kv_length·남은 decode 분포)을 in_flight_requests + DECODE 상태로 주입 + KV admit.
- [ ] 트레이스 포맷 불변(`arrival_time,prompt_len,max_tokens`). seed 는 *초기상태 확장*이지
  워크로드 표현 변경 아님.
- [ ] **caveat:** seed 분포가 현실 정상상태(ctx·decode 진행도)를 대표해야 편향 없음.
  분포는 워크로드에서 유도 또는 cold-start 1회 관측치로 seed.

### 2.6 F3 / Instance B 모델링 — B-FFN 을 스케줄에 반영 (확정 = 방식 i)

> **결정(사용자 2026-06-01): B-FFN 을 telemetry 가 아니라 *스케줄*에 넣는다.** μ-batch
> 다수의 일차 이유(F3)를 측정으로 입증하려면 A→B→A 종속이 실제 스케줄에 걸려야 함.

- [ ] **방식 (i) — 단일 이벤트루프에 B 를 자원으로 추가** (PIM 패턴 복제):
  - `node.py` — `NodeType.FFN` 추가 (μ-batch 당 5노드).
  - `dag.py` — 종속 `O_PROJ → FFN` 추가. (per-layer reset 그대로.)
  - `dispatcher.py` — 자원 `INSTANCE_B` + `instance_b_busy` + `pick_instance_b` +
    `dispatch_instance_b`(timing = `compute_ffn_op_time_s`, 이미 [config.py:246] 존재;
    `gpu_instance_b` 활동 기록). I4/I5 옆에 I6(B 동시 1 FFN) 추가.
  - `main_loop.py` — **layer advance 트리거를 O_PROJ 완료 → FFN 완료로 이동**
    (`_maybe_advance_forward_pass`). 다음 layer QKV 는 FFN 완료 후 dispatch.
  - 기존 `instance_pipeline.dispatch`(telemetry 기록)는 FFN 노드로 대체 → 정리.
- [ ] **F3 발현(자동):** DAG 에 μ-batch 간 edge 없음 → `FFN(M)` 이 INSTANCE_B 점유 중
  GPU 가 `QKV/attn(N)` backfill. 별도 정책 불필요(dispatcher.tick 이 3자원 각각 채움).
- [ ] **handoff:** NVLink A↔B 는 async hidden(ARCH §3.4·§3.5.3) → `max(A,B)` 산식에
  미포함. FFN 노드 timing 만 모델링.
- [ ] **ARCH 프레이밍 주석:** §6 DAG="Instance A only 4노드"는 문서상 범위 — FFN 노드에
  "inter-AB(F3) 스테이지, §3.4" 명시. [config.py:252] 주석("FFN 은 DAG 노드 아님") 갱신.
- [ ] **테스트 영향 (큼):** 4노드 가정 테스트(`test_dag`·`test_dispatcher`·`test_invariants`
  ·`test_main_loop_*`·`test_forward_pass`·`test_acceptance_*`·`test_cross_module_*`) 대량
  갱신. layer advance 트리거 이동(O_PROJ→FFN)도 반영.

---

## 3. 구현 체크리스트 (모듈별, surgical — 변경마다 즉시 단위테스트)

> 원칙(CLAUDE.md §3·§5): 요청과 무관한 줄 건드리지 않음. 한 모듈+테스트씩, 즉시 회귀.
> 변경 즉시 커밋. 무거운 측정은 백그라운드 하나씩.

- [x] **S0. F3 / Instance B 스케줄 모델링** (§2.6 방식 i) — 완료. `NodeType.FFN`(node.py),
  `O_PROJ→FFN` edge(dag.py), `INSTANCE_B` 자원 + `pick_instance_b`/`dispatch_instance_b`/
  I6(dispatcher.py), layer advance 트리거 O_PROJ→FFN(main_loop.py), `check_I6`(invariants.py).
  신규 테스트 `tests/test_phase2_ffn_stage.py` 9개 통과(DAG 5노드·FFN ready=O_PROJ done·
  dispatch·I6·단일 layer FFN 종료·F3 overlap[FFN(M) B점유 중 GPU 가 QKV(N) backfill]).
  미정리: 옛 `instance_pipeline.dispatch` telemetry 경로(이제 FFN 노드가 대체) + 4노드
  가정 옛 테스트(test_dag/test_dispatcher/test_invariants 등) — S2 후 일괄 갱신.
- [ ] **S1. `admission.py` 슬림화** → `balance_intra_A` 삭제, `balance_inter_AB` 처리
  결정 반영, `layer1` 에서 per-mb KV 예산·intra_A 제거. 단위테스트 갱신.
- [ ] **S2. `main_loop.py` 풀 former** → `_try_join`·`_per_mb_kv_budget`·`_recompose_mb`
  삭제, ADMISSION_TICK 핸들러를 per-iteration 배치 former 로 교체. L 도달 token 생성·전이
  유지, 그 뒤 풀 재선택.
- [ ] **S3. window/F2 정합** → former 가 활성 슬롯 2개를 유지해 F2 발현. capacity=3 유지.
  disjoint 분할 추적(요청→슬롯 매핑) 추가. 코드 변경 최소.
- [ ] **S4. 측정 substrate** → `Request.first_token_time` 추가, 완료 요청 sink(evaluator
  또는 scheduler 에 `completed_requests` list), L 도달 첫 decode 시 first_token_time 기록.
- [ ] **S5. 데드코드 정리** → S2 가 만든 orphan(import·필드·micro_batch.prefill_chunk_budget
  등) 제거. 사전존재 데드코드는 *언급만*.
- [ ] 자기리뷰 체크(커밋 전): 옛 이름 grep 0 / `python -c "import puls_sched"` 순환 0 /
  모듈 LOC 천장 / 플랜 인벤토리 vs 실제 diff 일치 / 미사용 import·죽은 분기 0.

---

## 4. 테스트 & 검증 (모듈별 + 메타 + 음성 + 교차불변)

> 기존 컨벤션: `tests/test_*.py`, `default_dummy_config()` + `_mk_*` 팩토리, 모듈별 단위.
> CLAUDE.md §5 — "green" ≠ "correct". 아래 4종 커버.

- [ ] **단위(모듈별):**
  - former: decode 전부 선택 + KV 여유만큼 prefill, cap 초과 시 큐잉.
  - balance(시간): t_pim>t_gpu → chunk>0 / t_pim≤t_gpu → chunk=0(슬랙 없음).
  - 전이: prefill 소진 → DECODE → 다음 iteration decode-set 포함.
- [ ] **메타-테스트(플랜 정합):** 삭제 대상 심볼(`_try_join`·`balance_intra_A`·
  `_per_mb_kv_budget`) 부재 단언 / NodeType·RequestState enum 완전성 / DAG edge=I1~I3.
- [ ] **음성 파라트라이즈(enum 교차곱):** RequestState 전이 cross-product — 불법 전이
  전수 raise. former 입력 경계(빈 풀/cap=0/KV 0/prefill만/decode만).
- [ ] **교차모듈 불변:** 임의 admit/evict/round-trip 후 (i) KVAccountant.used == Σ
  in_flight kv, (ii) decode-set ∩ prefill-queue = ∅, (iii) I4/I5(GPU·PIM 동시 1 op),
  (iv) 완료 요청 KV release 누락 0.
- [ ] **회귀:** 개발 중 가벼운 타깃(`pytest tests/test_admission*.py -q`), 커밋 직전 풀
  1회(`PYTHONIOENCODING=utf-8 pytest -q`). 깨진 기존 테스트는 풀 모델에 맞게 갱신(이유
  주석).

---

## 5. 측정 & 트레이스 (TBT·TTFT 중심)

- [ ] **measure 하네스 재작성**(`debug_phase2/measure_steady.py`): 스텁 해소.
  - 완료 요청 sink 에서 수집 → 정상상태 윈도우(warmup/cooldown 배제)만 절단.
  - **TTFT** = `first_token_time − arrival_time`. **TBT** = `(completion_time −
    first_token_time)/(decoded_count−1)`. 분포(p50/p90/max) 출력.
  - 보조: GPU/PIM 유휴(진단용), pipeline_efficiency.
- [ ] **warm-start seed init**(§2.5) 하네스 플래그 추가(`--seed-decode-pool`).
- [ ] **트레이스 = 기존 generator 재사용**(포맷 불변):
  - `gen_step5_traces.py` — ts(저ctx 128~1K)·tgen(중ctx 2~8K)·agentic(고ctx 32~128K).
  - 실행(§9 습관 준수, 백그라운드 하나씩):
    `PYTHONIOENCODING=utf-8 python debug_phase2/measure_steady.py --trace data/agentic_30.csv
    --seed-decode-pool ... > debug_phase2/out_agentic.txt`
- [ ] **검증 기대치(풀 모델 가설):**
  - 고ctx(agentic): prefill 이 PIM 슬랙에 숨어 **TBT≈t_pim 바운드 유지**(STEP5 의 TBT
    폭증 재현 안 됨이 성공 신호).
  - 저~중ctx(ts/tgen): GPU-bound·PIM 유휴 *정상*(물리 함수 ctx/56K). TBT 안정.
  - cross-check: `diag_optime.py`(Phase-2 판)로 op-time 직접 산출 = 하네스 t_pim/t_gpu 와 일치.

---

## 6. 문서 갱신 (구현·측정 후)

- [ ] [배치_생애.md](../../배치_생애.md) — 확정된 멤버십 정책(가/나) 반영.
- [ ] `debug_phase2/REPORT.md` 신규 — Phase-2 측정·TBT/TTFT 결과·Phase-1 대비.
- [ ] [README.md](../../README.md) "Target Workload" = long-context agentic 재프레이밍.

---

## 7. 작업 습관 / 리스크 (STEP6 §9 + 이번 세션 실측)

- [ ] **도구 과병렬 금지** — 이번 세션 cascade 취소 1회(§9 확인). 무거운 측정은 백그라운드
  하나씩.
- [ ] **Bash 의 한글 파일목록 출력이 깨짐**(artifact) — 디렉터리 나열은 Glob/Read 로.
- [ ] **단위 ns/µs** — PIM `op_time`=ns, clock/GPU=µs. 새 시간 비교마다 ×1e-3 확인
  (STEP5 버그 재발 방지, REPORT §13).
- [ ] **추측 말고 코드/측정** — op-time 은 diag_optime 로 직접.
- [ ] **변경 즉시 커밋**, 커밋 메시지는 heredoc, `PYTHONIOENCODING=utf-8` + 파일 출력.

---

## 진행 로그

- 2026-06-01: 정독 완료(배치_생애·ARCH §5.6/§6·REPORT §12~14·src 전모듈·measure 인프라).
  풀 모델 해석 확정(§0). 핵심: 기판은 이미 풀 모델, main_loop+admission 레이어만 교체.
- 2026-06-01: **멤버십 = (나) 디코더 분할 확정**(사용자). 결과로 §1b per-mb KV budget
  "삭제"→"재해석"(disjoint 분할 규칙 필요), 새 불변식(요청 동시 1슬롯), former 가 활성
  슬롯 2개 유지.
- 2026-06-01: **μ-batch 다수의 존재 이유 = F3(일차) 명확화**(사용자). B FFN 도는 동안 A
  안 쉬게 함. §2.3 재작성. **기판 공백 발견: B FFN 이 DAG/스케줄 종속 아닌 telemetry 만**
  (main_loop:330).
- 2026-06-01: **F3/B-FFN 을 처음부터 스케줄에 모델링 결정**(사용자 "어차피 설계할거면
  미리"). 방식 (i) 단일 이벤트루프 + INSTANCE_B 자원(§2.6). 재설계 범위가 기판(node·dag·
  dispatcher)까지 확장. **구현 순서 변경: S0(F3 모델링) → S1(admission) → S2(former) → …**
  S0 가 former 의 토대라 먼저. 다음: S0 착수.
- 2026-06-01: 가속 커버리지 확인(§0.6) — 동역학 2(더블버퍼링·인스턴스A∥B), op-time 2
  (F1·F5), closed-form 2(Aux). F2→"더블 버퍼링" 용어 통일. **분할 작업: 1차 S0~S2,
  2차 S3~S5** (사용자). 옛 전체 회귀 안 돎(다 교체) — 변경별 타깃 테스트만. test_invariants.py
  는 깨진 placeholder(마크다운 쓰레기 혼입) — 어차피 재작성.
