# PULS Scheduler Modules

`implementation/src/puls_sched/` 의 각 모듈 한 줄 역할 설명 (학습용). 각 Impl 완료 직전 (commit 직전 Self-review 동시 영역) 에 신설 / 변경 모듈 반영.

## Impl-1 — Core Data Structures + Event Loop Skeleton (commit `943eca5`)

- `config.py` — 시뮬레이션 전체에서 쓸 파라미터 (모델 구조 · 하드웨어 · 시간 · SLO · seed) 를 한 곳에 보관합니다.
- `clock.py` — 시뮬레이션의 현재 시각을 들고 있습니다. 시간을 앞으로만 흐르게 강제합니다.
- `request.py` — 요청 하나의 정보 (id · 입력 토큰 · 출력 토큰 · KV 길이 · 도착 시각) 와 그 요청이 지금 어느 단계 (대기 → prefill → decode → 완료) 에 있는지를 추적합니다.
- `micro_batch.py` — 같이 처리될 여러 요청을 하나의 μ-batch 로 묶고, 그 안에서 prefill chunk 와 decode token 을 분리해 들고 있습니다.
- `node.py` — DAG 한 노드의 정보 (작업 종류 · 어느 μ-batch 의 작업인지 · 지금 어떤 상태인지) 를 표현합니다.
- `dag.py` — μ-batch 가 들어오면 그 안의 작업 4 개 (QKV · prefill-attn · decode-attn · O-proj) 와 그들 사이의 선후 관계 (I1·I2·I3) 를 자동으로 만들어 줍니다.
- `event.py` — 미래 시점에 울릴 *알람* 한 건. 언제 (timestamp) 무슨 일이 (커널 완료 · 요청 도착 · admission tick) 벌어질지만 적어둡니다. 알람 받고 무엇을 할지의 *내용물* 은 `main_loop._handle` 영역.
- `event_queue.py` — 알람들을 시각 순서대로 줄 세워 두고, 다음 알람을 꺼낼 때마다 시계를 그 알람의 시각으로 맞춥니다 (= 시간 점프).
- `window.py` — 현재 처리 중인 μ-batch 3 개만 들고 있다가, 새 μ-batch 가 들어오면 가장 오래된 것을 자동으로 내보냅니다.
- `main_loop.py` — 큐에서 다음 알람을 꺼내 처리하는 메인 루프. 알람을 받았을 때 무엇을 할지의 *내용물* 은 Impl-2 부터 채워집니다.

## Impl-2 — Invariants + DAG Dispatcher

- `invariants.py` — 스케줄러가 절대 어기면 안 되는 5 규칙 (I1·I2·I3 = 작업 선후 관계 / I4·I5 = GPU·PIM 자원이 동시에 한 일만 수행) 위반을 잡아내는 검사 함수들입니다. 위반 시 에러를 던져 dispatch 를 차단합니다.
- `dispatcher.py` — DAG 를 보고 *지금 시작 가능한 노드들* (선행 작업 다 끝난 노드들) 을 골라낸 뒤, 우선순위 (O-proj > prefill-attn > QKV) 와 자원별 (GPU / PIM) 큐로 나눠서 한 번에 한 노드씩 보냅니다. GPU·PIM 자원이 지금 일하고 있는지의 상태도 들고 있습니다.
- `main_loop.py` — *(의미 변경)* Impl-1 의 빈 skeleton 에서, KERNEL_COMPLETION 알람을 받으면 dispatcher 에게 *작업 완료* 를 알리고 dispatcher 가 다음 작업을 보내도록 위임하는 실 루프가 되었습니다.

## Impl-3 — Admission Controller + k_total Decision + Request Queue + KV Accounting

- `config.py` — *(변경)* `AdmissionConfig` dataclass 신설. admission 관련 placeholder (N_sat · ctx tier 경계 · deadband width · idle θ · KV capacity · queue capacity · k_total dial step/max) 를 한 곳에 모음.
- `request_queue.py` — 도착한 요청을 admission 이 꺼낼 때까지 줄 세워 두는 bounded FIFO. 가득 차면 push 가 `False` 반환 (overflow reject).
- `kv_accountant.py` — 전체 KV slot 잔여 capacity 를 추적합니다. admit 시 demand 검증 + 차감, completion 시 회수. 중복 admit 또는 미admit 된 req 의 release 는 fail-fast raise.
- `idle_telemetry.py` — GPU · PIM 각 자원의 active duration 을 누적해 idle fraction 을 산출합니다. admission balance 결정의 입력. (Instance A · B 분리는 Impl-5)
- `deadband.py` — ctx 토큰 수 기반 tier (short / mid / long) 로 deadband width 를 lookup + |diff| ≤ width 의 in-band 판정. 값은 placeholder (ordering property 만 보존), 로직만.
- `k_total.py` — `kTotalDecider.solve` — 9-step dial {0, 256, ..., 2048} 에서 `t_PIM(k, N) ≤ t_proj` 를 만족하는 *최대* k 선택. 무 feasible 시 `over_budget=True` signal (admission escalation 트리거).
- `admission.py` — `Admission.layer1` 이 RequestQueue 에서 KV capacity 허용 한도까지 req 를 모아 → balance (inter-AB · intra-A) → mfu_floor clamp → k_total 결정 → `MicroBatchSpec` 산출. ARCH §6.4 admission 표 정합.
- `main_loop.py` — *(의미 변경)* `_handle` 의 `REQUEST_ARRIVAL` (RequestQueue push), `ADMISSION_TICK` (admission.layer1 호출 → 결과 spec 으로 InFlightWindow.admit → Dispatcher.tick) case body 채움. `SchedulerCore` 에 `request_queue` · `kv_accountant` · `admission` 멤버 + `_next_mb_id` allocator 추가.

## Impl-4 — PIM Executor Emulator

- `config.py` — *(변경)* PIM 관련 placeholder 3 개 추가. KV 캐시를 FP8 로 저장할지 FP16 로 저장할지 (모델 단위 시스템 설정), PIM 한 타일의 행 수 (RTL 합성으로 확정된 32 행), 그리고 8 GPU 가 같이 일할 때 추가로 드는 통신 시간을 보관합니다.
- `pim_emulator.py` — 한 번의 PIM 어텐션 연산이 *얼마나 걸리는지* 알려주는 시간 계산기입니다. "지금 PIM 으로 쓰는 채널 수" 와 "이번 batch 의 KV row 합" 두 가지만 주면 op time 을 즉시 산출해 줍니다. 채널 수가 한 GPU 의 채널 수보다 크면 (= 8 GPU 협력) 통신 시간도 더해 줍니다. Ramulator2 가 외부에서 미리 계산해 둔 cycle 데이터를 JSON 으로 읽어 들이는 로더도 같이 보유합니다 (실데이터 ingest 는 Impl-10).
- `dispatcher.py` — *(의미 변경)* PIM 노드 dispatch 시 *얼마나 걸릴지* 의 시간 산출 책임을 `pim_emulator` 에게 위임합니다. 이전엔 고정 lookup 한 값이었던 PIM op time 이, 이제 PIMExecutor 가 채널 수 · KV row 수 기반으로 계산해 돌려줍니다. 단 dispatch 시점에 그 두 정보가 어디서 오는지의 *진짜 흐름* 은 Impl-5 영역 — 지금은 config 의 placeholder 값을 그대로 씁니다.

## Impl-5 — Instance A/B Pipeline + Forward Pass + Inter-instance Handoff

- `instance.py` — Instance A 또는 B 의 *자원 추적기* 입니다. GPU 자원은 TP=8 lock-step (8 GPU 가 항상 같이 한 op 만 처리) 으로 1 단위, PIM 은 Instance A 만 보유. acquire/release 의 중복 점유 · 미점유 release 는 모두 raise 로 차단합니다.
- `nvlink.py` — Instance A ↔ B 의 NVLink 위 데이터 한 묶음 (decode `[B × hidden]` / prefill `[(B·chunk) × hidden]`) 이 건너가는 데 *얼마나 걸리는지* 만 알려주는 순수 시간 계산기입니다. 이벤트 push 도 자원 lock 도 하지 않습니다 — NVLink 은 데이터 통로 (dispatched resource 아님).
- `instance_pipeline.py` — 한 layer 의 A → handoff → B → handoff → A_next 흐름을 들고 있습니다. (1) Instance B 에 넘기는 텐서가 *항상 fixed shape* 인지 검사하고 (ragged 면 raise), (2) `steady_state_cycle(A_cycle, B_cycle) = max(A_cycle, B_cycle)` 의 ARCH literal 산식을 runtime 에 돌려줍니다. L-layer 루프는 보유하지 않습니다 (forward_pass 책임).
- `forward_pass.py` — *L 회 layer 통과* 의 loop owner 입니다. `LayerState.advance(mb)` 가 mb.current_layer_index 를 1 증가시키고, L 에 도달하면 *token decode signal* 의 trigger (True 반환) 를 돌려줍니다. 단조 위반 (역방향 · 이미 끝남) 은 raise. 실 instance_pipeline.dispatch 통합은 Impl-9 driver 영역입니다.
- `micro_batch.py` — *(변경)* admission 의 결정 정보를 dispatch 까지 운반하는 3 필드 신설: `k_total` (SP-PIM aggregate channel count), `kv_rows_total` (decode reqs 의 kv_length 합), `current_layer_index` (forward pass 의 현재 layer). Impl-4 의 carry-over O4.1 (dispatcher PIM 의 placeholder default args) 가 해소됩니다.
- `admission.py` — *(변경)* `MicroBatchSpec` 에 `kv_rows_total` 필드 추가. `layer1` 산출 시 decode_reqs 의 kv_length 를 합산하여 spec 에 담아 줍니다.
- `dispatcher.py` — *(의미 변경)* `micro_batches: dict[int, MicroBatch]` 멤버 + `register`/`unregister` API 신설. `_op_time` 의 PIM branch 가 이제 placeholder 가 아닌 *실제 mb.k_total · mb.kv_rows_total* 로 op_time 을 산출합니다 (O4.1 해소). 미등록 mb 의 PIM dispatch 는 raise.
- `main_loop.py` — *(의미 변경)* `ADMISSION_TICK` body 가 `MicroBatchSpec` → `MicroBatch` 변환 (Q1 — 변환 시점) 후 `dispatcher.register(mb)` 호출 → `window.admit(mb_id)` → `dispatcher.tick()` 의 순서로 진행합니다. spec 의 결정 정보 (k_total · kv_rows_total) 이 dispatcher 까지 유실 없이 흐릅니다.
