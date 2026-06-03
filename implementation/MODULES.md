# PULS 스케줄러 시뮬레이터 — 모듈 맵

`implementation/src/puls_sched/` 각 모듈의 **역할 · 기능 · 필요 이유**. (로컬 전용 작업 노트)

---

## 1. 코어 이벤트 / 디스패치 엔진

### event.py
- **역할**: 시뮬레이션 이벤트의 자료형 정의 모듈.
- **기능**:
  - `EventType` enum 으로 이벤트 종류(KERNEL_COMPLETION, REQUEST_ARRIVAL, ADMISSION_PASS)를 구분한다.
  - `Event` 데이터클래스가 `timestamp`, `type`, `payload`(dict) 를 묶어 큐가 다룰 단위로 표현한다.
- **필요 이유**: 스케줄러의 모든 동작을 시각이 붙은 이벤트로 통일해 처리하기 위한 공통 자료형이 필요하다.

### event_queue.py
- **역할**: 시각 순으로 이벤트를 꺼내는 우선순위 큐.
- **기능**:
  - `push` 가 `(timestamp, seq, event)` 튜플을 heap 에 넣되, `clock.now` 이전 시각의 이벤트는 거부한다.
  - `pop` 이 최소 시각 이벤트를 꺼내며 동시에 `clock.advance_to` 로 시계를 그 시각으로 전진시킨다.
  - 단조 증가 `_seq` 로 같은 시각 이벤트의 삽입 순서를 안정적으로 결정하고, `peek_timestamp` 로 다음 시각을 조회한다.
- **필요 이유**: 이산 사건 시뮬레이션에서 이벤트를 시간 순서대로 결정론적으로 처리하려면 시계와 결합된 정렬 큐가 필요하다.

### clock.py
- **역할**: 시뮬레이션 전역 시각을 보유하는 단조 시계.
- **기능**:
  - `now` 프로퍼티로 현재 시각을 노출한다.
  - `advance_to(t)` 가 시각을 전진시키되 과거로의 역행은 예외로 막아 단조성을 강제한다.
- **필요 이유**: 모든 모듈이 공유하는 단일 시간 기준점이 있어야 이벤트 순서와 op 종료 시각 계산이 일관되게 유지된다.

### dag.py
- **역할**: μ-batch 별 연산 노드와 선후 종속(precedence)을 담는 DAG.
- **기능**:
  - `add_micro_batch` 가 μ-batch 마다 QKV·PREFILL_ATTN·DECODE_ATTN·O_PROJ·FFN 노드를 생성하고 종속 관계(QKV→attn→O_PROJ→FFN)를 설정한다.
  - `reset_micro_batch` 가 노드들을 fresh PENDING 으로 재생성해 다음 layer 의 재실행 진입점을 만든다(종속은 불변).
  - `remove_micro_batch`/`get_node` 로 μ-batch 제거와 개별 노드 조회를 제공한다.
- **필요 이유**: forward pass 의 연산 간 선후 관계를 명시해, dispatcher 가 어떤 노드를 언제 실행 가능한지 판정할 근거를 제공한다.

### dispatcher.py
- **역할**: 자원(GPU·PIM·Instance B)에 ready 노드를 배정하고 종료 이벤트를 발행하는 스케줄링 코어.
- **기능**:
  - `refresh_ready` 가 선행 노드가 모두 DONE 인 PENDING 노드를 READY 로 전이시키고, `pick_gpu`/`pick_pim`/`pick_instance_b` 가 우선순위(GPU: O_PROJ>PREFILL_ATTN>QKV)·oldest μ-batch 순으로 노드를 고른다.
  - `dispatch_*` 가 종속(I1/I2/I3)·자원 busy(I4/I5/I6) 불변식을 검사한 뒤 노드를 RUNNING 으로 바꾸고 `_op_time` 으로 산출한 종료 시각에 KERNEL_COMPLETION 이벤트를 push 한다.
  - `_op_time` 이 μ-batch spec 기반으로 GPU/FFN/PIM op 시간을 계산하며, `tick` 이 비어 있는 자원마다 한 번에 한 노드씩 디스패치한다.
  - `register`/`unregister` 로 μ-batch lookup 을 관리하고, `on_dispatch` 콜백으로 외부 inspector(evaluator)에 디스패치 시점을 통지한다.
- **필요 이유**: 종속·자원 제약을 지키면서 연산을 실제 하드웨어 자원에 매핑하고 그 소요 시간을 시뮬레이션 이벤트로 변환하는 핵심 실행 주체가 필요하다.

### invariants.py
- **역할**: 디스패치 시점에 검사하는 정합성·자원 불변식 모음.
- **기능**:
  - `check_I1`/`check_I2` 가 prefill·decode attention 디스패치 전 해당 μ-batch 의 QKV 완료를 강제한다.
  - `check_I3` 가 O_PROJ 디스패치 전 prefill·decode attention 둘 다 완료됐는지 검사한다.
  - `check_I4`/`check_I5`/`check_I6` 가 GPU·PIM·Instance B 각 자원이 동시에 한 op 만 실행하도록 busy 플래그를 검증한다.
- **필요 이유**: 잘못된 종속/자원 위반을 즉시 예외로 드러내, 스케줄링 로직의 정확성을 실행 중에 보장하기 위함이다.

### main_loop.py
- **역할**: 이벤트 처리·admission·μ-batch 구성을 묶는 스케줄러 최상위 루프(`SchedulerCore`).
- **기능**:
  - `step`/`run_until_empty` 이 큐에서 이벤트를 꺼내 `_handle` 로 분기하며, KERNEL_COMPLETION·REQUEST_ARRIVAL·ADMISSION_PASS 세 종류를 처리한다.
  - `_refill_pool` 이 KV 게이트만으로 request_queue 를 in_flight 풀로 들이고, `_fill_window`/`_compose_microbatch` 가 decode-set 과 prefill 청크를 독립 구성해 활성 μ-batch 를 staggering 목표(2)까지 채운다.
  - `_maybe_advance_forward_pass` 가 FFN 완료를 layer 경계로 보고 `LayerState.advance`, 토큰 디코드 신호, 요청 상태 전이(PREFILL→DECODE), 완료 요청 finalize 를 수행한 뒤 μ-batch 를 재구성하거나 evict 한다.
  - `_populate_mb_phases` 가 prefill 토큰을 멤버에 그리디 분배하며 depth-합을 동작점 목표로 수렴시키고 age-cap 으로 starvation 을 막는다.
- **필요 이유**: 개별 모듈(큐·DAG·dispatcher·admission)을 하나의 이벤트 구동 실행 흐름으로 통합해 요청 수명주기와 연속 스케줄링을 진행시키는 중심 오케스트레이터가 필요하다.

---

## 2. Admission / μ-batch 구성

### admission.py
- **역할**: in-flight DECODE 풀에서 한 μ-batch 의 decode-set 을 선택하는 admission 정책.
- **기능**:
  - `steer_decode_set(candidates)` 로 풀에서 개수 타깃(`decode_count_target`)과 Σkv 타깃(`kv_operating_target`)을 동시에 맞추도록 그리디 선택한다.
  - 매 선택마다 `ideal=(target_kv−S)/(target_count−n)` 에 `kv_length` 가 가장 가까운 디코더를 고르는 steering 을 적용한다.
  - `wait ≥ age_cap` 인 디코더가 있으면 가장 오래 기다린 것을 강제 선택해 starvation 을 막는다.
  - 선택분은 `wait=0` 리셋, 미선택분은 `wait+=1` 로 다음 차례 후보로 남긴다.
- **필요 이유**: KV 예산과 배치 크기를 동시에 만족시키면서 공정성까지 보장하는 decode 선택 로직이 필요하다.

### kv_accountant.py
- **역할**: KV 캐시 용량을 추적하고 admit/release 를 회계 처리하는 모듈.
- **기능**:
  - `can_admit(req)` 로 `req.kv_length` 가 잔여 용량 이내인지 검사한다.
  - `admit(req)` 로 사용량을 늘리고 요청별 점유량을 기록하며, 중복 admit·용량 초과 시 예외를 던진다.
  - `release(req)` 로 점유량을 회수하고, 미admit 요청 release 시 예외를 던진다.
  - `capacity`/`used`/`remaining` 프로퍼티로 현재 상태를 노출한다.
- **필요 이유**: KV 메모리 오버플로와 이중 점유를 막는 단일 회계 기준점이 필요하다.

### micro_batch.py
- **역할**: 한 μ-batch 의 prefill/decode 작업과 실행 메타데이터를 담는 데이터 컨테이너.
- **기능**:
  - `prefill_chunk`(요청별 토큰)와 `decode_tokens`(요청별 디코드)를 보관한다.
  - `kv_rows_total`, `prefill_processed` 등 admission 산출값과 forward-pass 입력을 저장한다.
  - `current_layer_index` 로 L-iteration 현재 위치를 추적하고, `request_ids()` 로 구성 요청을 조회한다.
- **필요 이유**: 스케줄러가 만든 배치 구성과 실행기가 쓰는 런타임 상태를 한 객체로 운반하기 위해서다.

### window.py
- **역할**: 동시 in-flight μ-batch 개수를 고정 천장으로 제한하는 슬라이딩 윈도우.
- **기능**:
  - `capacity`(기본 3 = 2 active + 1 전이 여유)만큼 μ-batch id 를 `deque` 로 유지하며, F2 ablation 설정으로 capacity 를 덮어쓸 수 있다.
  - `admit(mb_id)` 로 윈도우에 추가하고 가득 차면 가장 오래된 것을 DAG 에서 제거·반환한다.
  - `evict(mb_id)` 로 특정 μ-batch 를 명시적으로 제거하며 DAG 에서도 함께 떼어낸다.
  - `current_ids()` 로 현재 윈도우 내용을 노출한다.
- **필요 이유**: staggering 깊이를 작게 고정해 동시 KV 점유가 캐시를 독점하는 것을 막기 위해서다.

### completion.py
- **역할**: 요청 종료를 검출하고 KV 슬롯을 회수하는 lifecycle 종료 처리기.
- **기능**:
  - `check(req, eos_seen)` 로 COMPLETED 상태·EOS·`decoded_count ≥ max_tokens` 종료 조건을 idempotent 하게 판정한다.
  - `finalize(req)` 로 KV release 를 먼저 수행한 뒤 `completion_time` 을 기록하고 COMPLETED 로 전이한다.
  - 중복 finalize 나 admission 이전(PENDING) finalize 는 예외로 막는다.
- **필요 이유**: 종료된 요청의 KV 를 즉시 회수해 메모리를 풀고 상태 전이를 일관되게 보장하기 위해서다.

### request.py
- **역할**: 단일 요청의 상태와 lifecycle 데이터를 보유하는 핵심 도메인 객체.
- **기능**:
  - `RequestState`(PENDING/PREFILL/DECODE/COMPLETED) enum 과 전이표로 허용된 전이만 정의하고, `transition_to` 가 잘못된 전이를 예외로 거부한다.
  - `prompt_len`, `kv_length`, `decoded_count`, `prefill_processed` 등 스케줄링·진행 상태를 담는다(토큰 내용 대신 길이만 보존).
  - `arrival_time`/`first_token_time`/`completion_time`, `wait`/`prefill_wait` 로 측정값과 공정성 추적을 보관한다.
- **필요 이유**: 요청별 상태를 한 곳에서 소유하며 불법 전이를 막는 단일 진실 원천이 필요하다.

### request_queue.py
- **역할**: 도착했지만 아직 admit 되지 않은 대기 요청을 담는 용량 제한 FIFO 큐.
- **기능**:
  - `push(req)` 로 뒤에 추가하되 capacity 초과 시 `False` 를 반환해 거부한다.
  - `pop_oldest()` 로 가장 오래된 요청을 꺼내고, 비었으면 `None` 을 반환한다.
  - `peek_oldest()` 로 제거 없이 선두를 확인하고, `__len__` 으로 길이를 노출한다.
- **필요 이유**: admission 이전 요청을 도착 순서대로 보관하면서 큐 폭주를 용량으로 제어하기 위해서다.

---

## 3. Substrate / 시간 모델

### config.py
- **역할**: 모델/하드웨어/캘리브레이션 상수와 op 별 시간 산출 공식을 모은 단일 설정 모듈.
- **기능**:
  - `ModelConfig`/`HWConfig`/`SLOConfig`/`AdmissionConfig`/`CalibrationConfig` 등 frozen dataclass 로 모델 스펙·GPU/HBM 캘리브레이션 값·SLO·동작점 타깃을 보관한다.
  - `compute_gpu_op_time_s` 가 QKV·PREFILL_ATTN·O_PROJ 각 노드의 FLOPs 를 batch·ctx 로 계산해 `FLOPs / (peak×MFU×num_gpus)` 로 op 시간을 산출한다.
  - `compute_ffn_op_time_s` 가 Instance B SwiGLU FFN 의 op 시간을 동일 방식으로 계산한다.
  - `default_dummy_config` 가 Llama-3 70B 급 캘리브레이션 값을 주입한 완성 Config 를 만든다.
- **필요 이유**: 모든 모듈이 공유하는 하드웨어 상수와 시간 모델의 단일 진실 공급원이기 때문이다.

### pim_emulator.py
- **역할**: SP-PIM 의 op 시간을 계산하는 stateless 시간 계산기.
- **기능**:
  - `tile_time` 이 `kv_precision`(FP8/FP16) regime 에 따라 타일 단위 시간을 룩업한다.
  - `k_aggregate` 가 Instance A 전체 PIM 채널 수(GPU×stack×channel)를 산출한다.
  - `op_time` 이 KV row 합을 채널×타일행으로 나눠 채널당 타일 수를 올림 계산하고 `tile×tile_time + cross-GPU broadcast` 로 op 시간을 반환한다(F5 비활성 시 lock-step penalty 경로).
  - `load_ramulator2_cycles` 가 Ramulator2 cycle JSON 을 fail-fast 검증하며 ingest 한다.
- **필요 이유**: PIM 이 처리하는 decode-attention 의 시간을 아키텍처 스펙대로 산출해 사이클 모델에 공급하기 위해서다.

### forward_pass.py
- **역할**: μ-batch 를 L 개 레이어에 걸쳐 반복시키는 L-loop 소유자.
- **기능**:
  - `LayerState.advance` 가 `current_layer_index` 를 단조 증가시키고 L 도달 시 토큰 디코드 신호로 True 를 반환한다.
  - `ForwardPass.run` 이 매 레이어마다 instance pipeline dispatch 를 호출해 A→B→A_next 체인을 엮으며 완료된 레이어 수를 반환한다.
- **필요 이유**: "forward pass = L × cycle" 구조에서 레이어 반복 제어와 단일 레이어 사이클 처리를 분리하기 위해서다.

### instance.py
- **역할**: Instance A/B 의 GPU·PIM 자원 점유 상태를 추적하는 리소스 트래커.
- **기능**:
  - `acquire_gpu`/`release_gpu` 가 GPU busy 플래그를 설정/해제하며 중복 점유·이중 해제를 예외로 막는다.
  - `acquire_pim`/`release_pim` 이 PIM 을 동일하게 다루되, PIM 없는 Instance B 호출 시 예외를 던진다.
- **필요 이유**: 두 인스턴스의 자원 동시 점유 충돌을 방지하는 단순 상태 가드가 필요하다.

### instance_pipeline.py
- **역할**: A→handoff→B→handoff→A_next 단일 레이어 사이클 소유자.
- **기능**:
  - `validate_handoff_shape` 가 A→B 핸드오프 텐서가 `[tokens × hidden]` 고정 형태인지, prefill chunk 길이가 균일한지 검증한다.
  - `steady_state_cycle` 이 정상상태 사이클을 `max(A_cycle, B_cycle)` 로 반환한다(NVLink 제외).
  - `dispatch` 가 매 레이어마다 핸드오프를 검증하고 `compute_ffn_op_time_s` 로 Instance B FFN 시간을 산출해 `gpu_instance_b` 활동 구간을 텔레메트리에 기록한다.
- **필요 이유**: 인스턴스 간 파이프라인 한 사이클의 연결·검증·B측 활동 측정을 한 곳에서 담당하기 위해서다.

### nvlink.py
- **역할**: A↔B 인스턴스 간 NVLink 전송 시간을 계산하는 stateless 순수 함수.
- **기능**:
  - `time` 이 텐서 shape 의 전체 바이트 수에 `nvlink_time_per_byte_ns` 를 곱해 전송 시간을 반환하며, 빈 shape·음수 차원을 예외로 막는다.
- **필요 이유**: A→B·B→A 데이터 경로 비용을 표현하되, 이 시간은 비동기로 A/B 연산에 가려지므로 설계상 사이클 산식(max(A,B))에서 제외된다.

### node.py
- **역할**: DAG 노드 타입·상태와 상태 전이 규칙 정의 모듈.
- **기능**:
  - `NodeType` enum 이 QKV·PREFILL_ATTN·DECODE_ATTN·O_PROJ·FFN 노드 종류를 정의한다.
  - `NodeState` enum 과 전이표가 PENDING→READY→RUNNING→DONE 단방향 전이만 허용하고, `Node.transition_to` 가 위반을 예외로 차단한다.
- **필요 이유**: 스케줄러가 다루는 연산 노드의 종류와 합법적 생명주기를 한 곳에서 강제하기 위해서다.

### idle_telemetry.py
- **역할**: 3개 자원 슬롯(GPU-A·PIM·FFN)의 활동/유휴 시간을 측정하는 텔레메트리 트래커.
- **기능**:
  - `record_active` 가 자원별 활동 구간을 누적하고 윈도우 끝을 갱신하며, legacy "GPU"/"PIM" 키를 instance_a 슬롯으로 매핑한다.
  - `idle_fraction`/`active_duration` 이 윈도우 대비 유휴 비율과 누적 활동 시간을 반환한다(span≤0 가드).
  - `gpu_idle_fraction`/`pim_idle_fraction`/`gpu_instance_b_idle_fraction` alias 로 자원별 유휴 비율을 노출한다.
- **필요 이유**: intra-A(GPU vs PIM)·inter-AB 자원 균형 신호를 수집해 리포트/검증에 공급하기 위해서다.

---

## 4. 오케스트레이션 / 평가

### run.py
- **역할**: 시뮬레이터의 end-to-end 드라이버 — config·trace·모든 모듈을 조립해 스케줄러를 끝까지 돌리고 리포트를 산출한다.
- **기능**:
  - `init` 이 config factory 를 dotted-path 로 로드하고 Clock·Queue·DAG·Dispatcher·Admission·SchedulerCore 등 전 모듈을 인스턴스화·wiring 하며, trace 를 미리 읽어 REQUEST_ARRIVAL 이벤트로 push 하고 Evaluator 를 dispatch hook 에 연결한다.
  - `step`/`loop` 이 `SchedulerCore.step()` 을 위임 호출하며 큐·window·in_flight 가 모두 빌 때까지 반복하고, 무한루프 방지용 안전 step 한도를 둔다.
  - `teardown` 이 `evaluator.report()` 결과를 JSON·Markdown 파일로 출력한다.
  - `main` 이 argparse 로 인자를 파싱해 init→loop→teardown 을 실행하고 exit code 를 반환한다.
- **필요 이유**: 개별 모듈을 하나의 실행 가능한 시뮬레이션 파이프라인으로 묶는 진입점이자 조립 지점이기 때문이다.

### evaluator.py
- **역할**: 절대 성능지표가 아닌 **구조적 평가 리포트**(dispatch trace·idle fraction·PIM utilization·pipeline efficiency·가속 분해)를 산출하는 모듈.
- **기능**:
  - `record_dispatch` 콜백으로 dispatch 이벤트를 누적해 `dispatch_trace`·`idle_fraction`·`pim_utilization` 을 산출한다.
  - `pipeline_efficiency` 로 `max(A,B)/(A+B)` 비율을, `acceleration_decomposition` 으로 F1·F2·F3·F5 의 on/off cycle 비율(방향성)을 산출한다.
  - Aux1(mixed-batch weight 재사용)·Aux2(KV bus traffic 절감)·F3(closed-form)·F5(channel-independent vs lock-step) 산식과 measured 값의 cross-validate 를 제공한다.
  - `report` 로 dict + Markdown 표(provenance 라벨 동반)를 생성한다.
- **필요 이유**: 스케줄러 동작이 구조적으로 기대한 가속·균형 특성을 갖는지 증거로 제시하기 위해 필요하다.

### trace.py
- **역할**: long-ctx production trace(CSV)를 읽어 Request 로 재생하고 분포 통계를 내는 모듈.
- **기능**:
  - `load` 가 longbench CSV 스키마를 검증(헤더·타입·음수)하며 ingest 하고, 미지원 스키마는 명시적 예외를 던진다.
  - `replay` 가 각 entry 를 Request(prompt_len·kv_length·arrival·max_tokens)로 변환하는 generator 를 제공한다(다회 호출 일관성 보장).
  - `synthesize` 가 seed 기반 결정론(bit-exact) 합성 trace 를 생성한다.
  - `stats` 가 KV 길이·arrival interval 의 min/max/mean/std 분포 통계를 산출한다.
- **필요 이유**: 시뮬레이션 입력 워크로드를 외부 의존 없이 재현 가능하게 공급하기 위해 필요하다.

### __main__.py
- **역할**: `python -m puls_sched` 실행을 위한 패키지 진입점.
- **기능**: `Run.main(sys.argv[1:])` 을 호출하고 그 반환 exit code 로 프로세스를 종료한다.
- **필요 이유**: 패키지를 모듈 형태로 직접 실행할 수 있게 하기 위해 필요하다.

### __init__.py
- **역할**: `puls_sched` 패키지 마커(빈 파일).
- **기능**: 디렉터리를 임포트 가능한 패키지로 만든다.
- **필요 이유**: 하위 모듈을 `puls_sched.*` 로 임포트할 수 있게 하기 위해 필요하다.
