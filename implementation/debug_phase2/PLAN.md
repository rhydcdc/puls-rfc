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

> **Overlap ≠ Balance (사용자 정정 2026-06-01, §0.8 에서 구체화).** S0 가 준 것은
> *overlap*(B 도는 동안 A backfill, 동역학). *balance*(셋 시간을 맞춤)는 별개 축 —
> 최종적으로 **정적 동작점**(KV 합 25M + prefill 512, §0.8)으로 결정. 동적 cycle 측정·
> 유휴율 게이트는 불필요(§2.5). 유휴율은 진단 출력으로만.

> **메모리 = HBM4 (사용자 정정).** [config.py:112-116] — 2 TB/s/stack × 8 = 16 TB/s/GPU,
> 8 GPU 합산 128 TB/s (HBM4 hypothetical projection). 인스턴스 B FFN 은 compute-bound 라
> 시간 모델에 메모리 항 *고의 생략*(HBM/GDDR 에서 FFN memory-bound 확률 0 — 죽은 분기 회피).

## 0.7 ★ TP=8 분산 버그 수정 (2026-06-01) — 모든 균형 숫자의 토대

> **발견(사용자 지적):** `compute_gpu_op_time_s`·`compute_ffn_op_time_s` 가 단일 GPU
> peak(2200 TFLOPS *per GPU*)로만 나눠 GEMM wall-clock 을 **8배 과대평가**. PIM 은
> `k_aggregate`(8-GPU 채널 합산)로 분산 반영하는데 GPU/FFN 만 단일 → 단위 불일치.
> commit `40e812a` 수정 (num_gpus 파라미터, A=8·B=8).

**수정 후 — TP=8 의 직접 효과 (op-time 직접 산출). 정밀 동작점은 §0.8 가 최종.**

| 지표 (TP 버그 효과) | 버그(전, ÷1) | 수정(후, ÷8) |
|---|---|---|
| PIM = GPU-proj 임계 ctx | 56,160 | **~7,020** (÷8) |
| 256 토큰 FFN | 273 µs | **34 µs** |

- **PIM=GPU-proj 임계 7K**: 7K 토큰만 넘어도 PIM 이 GPU proj 추월 시작. long-context
  진입 장벽 8배 하락. GPU·B 가 8배 빨라져 PIM 이 의미 갖는 구간이 크게 넓어짐.
- **저ctx(<7K)**: PIM < GPU-proj (decode-attn 싸서 PIM 유휴 *정상*, ARCH §6.6). 동작점
  former 가 작은 배치로 흡수 — 별도 처리 불필요(§0.9).
- **★ 삼중 균형(PIM=GPU-A=B)의 정밀 동작점·spread·KV 캐파는 모두 §0.8 이 최종**
  (KV 합 25M/배치, prefill 512, spread 0.6%, 캐파 배치당 30M·총 60M). 여기엔 옛 prefill-free 추정치를 두지
  않음 — 충돌 방지. (§0.8 의 "평균 ctx ~100K" = 이 균형의 *요청 길이* 표현일 뿐.)

## 0.8 ★ 동작점(operating point) — KV 총량 기준 밸런스 (확정 2026-06-01)

> **스케줄러의 진짜 레버 = "배치에 넣는 디코더들의 KV 길이 합" = PIM KV 총량.**
> 개별 요청 ctx 는 트레이스가 줌(제어 불가). 스케줄러는 디코더를 *골라 담아* KV 합을
> 목표 범위에 맞춘다. **디코드는 넣기는 자유, 쪼개 넣기는 불가** (요청 KV 통째) → 정확히
> 한 점에 못 맞추므로 **오차 범위** 안에 들어오게 고른다 (사용자 핵심).

**세 시간 = 무엇의 함수 (단위 통일: 전부 토큰):**
- `t_PIM` = f(**KV 총량** = Σ 디코더 KV). 개별 ctx 분포 무관, *합만* 중요.
- `t_GPU-A` = QKV·O proj(batch=N_dec+prefill) + PREFILL_ATTN(prefill × 그 요청 ctx).
- `t_B(FFN)` = f(batch = N_dec + prefill).

**확정 동작점 (prefill=512 고정, op-time 직접 산출, TP=8 반영):**

| 파라미터 | 값 |
|---|---|
| **목표 KV 총량** | **25,000K** (= 25M 토큰) → PIM≈GPU-A≈B≈101µs, spread **0.6%** |
| **허용 범위 (15% 오차)** | **21,500K ~ 29,000K** |
| prefill (배치당) | **512 토큰** 고정 (2^9, vLLM `max_num_batched_tokens` 동급) |
| 균형 시간 X | ~101 µs |
| N_dec (디코더 수) | *부산물* — 평균 ctx 100K면 ~248개, 50K면 ~500개 |

- **PIM 시간 표**(검증): KV 20M→81.7µs / 25M→102µs / 30M→122µs (개별 ctx 무관, 합만).
- **15% = KV 총량 21.5M~29M** = (평균 ctx 100K 가정 시) 요청 길이 ~87K~117K 에 해당.
- 그 밖: KV<21.5M → PIM 작아 GPU/B 그늘(놂). KV>29M → PIM bottleneck(A-bound). 둘 다
  레버로 못 고침 — 풀에 그 KV 합을 만들 디코더가 없으면(짧은 요청만) 균형 불가 = 정상.

**KV 캐파 = 배치당 30M, 총 60M aggregate (확정 2026-06-01, 사용자 A안).** 동작점 25M·상한
29M 은 *마이크로 배치 하나*의 KV 합 → 배치당 천장 30M (former 가 21.5~29M 로 admit, OOM 방지
여유). **마이크로 배치 2개가 A∥B(F3) 오버랩을 위해 동시 in-flight** 이고 디코더는 둘에
disjoint·영구 상주(§3.3) → 총 상주 KV = **2 × 30M = 60M aggregate**.
- HW = **160GB/stack × 64 stack = 10.24 TB** (= 80GB×64 의 2배). 가중치 137GB 제외 후 10.1 TB
  / 163,840 B per token = **61.7M 토큰** → 60M 담고 여유. (옛 "80GB/64stack 5TB → 30M aggregate"
  는 배치 *하나*치만 담겨 A∥B 오버랩 불가였음 — 사용자 정정.)
- 코드 정합: `_per_mb_kv_budget = kv_capacity_aggregate / _STAGGERING_TARGET_MB(2)` = **60M/2 =
  30M per micro-batch** → 배치당 천장과 자동 일치. config `kv_capacity_aggregate` = **60M** 으로
  설정(S2/S5). per-mb 예산 cap/2 는 "땜질"이 아니라 *60M 총량의 2-슬롯 disjoint 분할*로 정합.

**타깃 워크로드 = long-context agentic, 요청 ctx ~87K~117K (중심 100K).** README
"Target Workload" 이 범위로 재프레이밍. 그 밖 ctx 는 균형 덜 맞음 = PULS 적용 범위 밖
(하드웨어 물리, ARCH §6.6 정합).

## 0.9 짧은 컨텍스트 — 동작점이 흡수, 별도 정책 불필요 (사용자 확정 2026-06-01)

> 초기엔 "짧은 요청 골라 빨리 쳐내는" length-aware 정책을 검토했으나 **불필요로 결론.**
> §0.8 동작점 former("KV 합을 [21.5M,29M] 까지 채우고 prefill 512")가 이미 length 를
> 암묵 처리: 짧은 요청만 있는 풀은 KV 합이 25M 에 못 미쳐 작은 배치가 되고 FFN 도 작아
> 빨리 끝남(인위적 선별 불필요). 그때 **PIM 이 노는 건 고칠 대상이 아니라 물리적 정상**
> (저ctx decode-attn 이 싸서, ARCH §6.6) — 그래서 PULS 타깃이 long-context(§0.8). 즉
> "짧은 것 고르기" 레버는 동작점 하나로 흡수되어 사라짐.

## 1. 코드 인벤토리 — 재사용(substrate) vs 교체(scheduling)

### 1a. 재사용하는 기판 (substrate — 대부분 무수정, F3 확장만 추가)

> ⚠ `node.py`·`dag.py`·`dispatcher.py`·`main_loop` 의 layer advance 는 **S0 에서 F3
> 확장**(FFN 노드·INSTANCE_B 자원, §2.7). 그 외 골격은 그대로.

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
- [x] `main_loop.py::_join_gate_open` + `_try_join` + hysteresis 게이트 — **삭제 완료(S2)**.
  `_backfill_slot`(유휴 게이트 없음)으로 대체.
- [ ] `main_loop.py::_compose_admission_payload`·`_measure_cycles`·`_make_t_pim_fn`·
  `_prev_a/b_active_snapshot` — **삭제(§2.5).** 동작점 고정이라 매 tick cycle 측정 불필요.
  ADMISSION_TICK payload 가 trivial 화 (동작점 former 는 KV 합·prefill 512 만 봄).
- [ ] `main_loop.py::_recompose_mb` (373~393) — **교체.** "같은 req 집합 재구성+합류" →
  "풀에서 per-iteration 재선택".
- [ ] `main_loop.py::_maybe_advance_forward_pass` (307~371) 의 recompose/evict 꼬리
  (360~371) — **교체.** L 도달 시 token 생성·상태전이(337~358)는 *유지*, 그 뒤 재구성
  로직만 풀 모델로.
- [ ] `main_loop.py::ADMISSION_TICK` 핸들러 (146~187) 의 "spec→영속 mb 1개 생성" —
  **교체.** per-iteration 배치 former 로.
- [x] `admission.py::balance_intra_A` — **삭제 완료(S1).** 유일한 유휴율 기반 레버였음.
- [ ] `admission.py::balance_inter_AB`·`balance_pim_slack` — **둘 다 삭제(§2.5).** 동작점이
  정적 고정(KV 합 25M + prefill 512)이라 동적 cycle 측정·prefill 사이징 불필요. deadband
  (`deadband.py`)도 같이 moot. (S1 때는 inter_AB 유지로 봤으나 §0.8 동작점 확정 후 삭제로 변경.)
- [ ] `admission.py::layer1` (98~176) — **재작성.** head-of-line walk + KV admit 은 유지
  (멤버십=용량). 단 admit 종료 조건 = **Σkv 가 [21.5M,29M] 도달** + prefill 512 고정.
  t_proj·t_pim_fn·a_cycle·b_cycle·per_token·balance_* 인자/호출 전부 제거.
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

### 2.0 멤버십 = 디코더 분할 (확정 = 나)

- [x] **결정:** 풀의 decoder 를 **2개(목표) in-flight μ-batch 로 disjoint 분할**, window
  3번째 슬롯은 전이 여유. 연속 μ-batch 가 서로 다른 요청을 담아 더블 버퍼링·F3 overlap
  (PIM 이 M decode-attn 하는 동안 GPU 가 M+1 QKV; B 가 M FFN 하는 동안 A 가 N attention).
  ARCH §5.6·§6.5·§6.7 정합.
- [x] **재충전 = sticky 슬롯 + 빈자리 backfill (§2.2 확정).** 슬롯 decoder 유지 + 완료분
  제거 + 풀에서 신규 충전. disjoint 유지 단순, KV 안정.
- [x] **TBT 정의:** = 한 decoder 가 자기 슬롯의 forward pass 한 바퀴(연속 두 토큰 간격).
  슬롯들이 겹쳐 돌아(F3/더블버퍼링) throughput↑, 동작점 유지 시 슬롯 cycle 불변 → TBT 불변.

### 2.1 자료구조 — 전역 풀

- [ ] **decode-set** = `in_flight_requests` 중 `state==DECODE` (이미 prefill 끝, 매 step
  1토큰). substrate 의 `in_flight_requests` dict 재활용.
- [ ] **prefill-queue** = `state∈{PENDING,PREFILL}` 이며 `prefill_processed <
  len(prompt_tokens)` (남은 프롬프트 chunk 필요). `request_queue`(미admit) + in_flight 의
  prefill 진행 요청.
- [ ] 한 요청은 prefill 소진 시 decode-set 으로 *전환*(빠지는 게 아님). `request.py` FSM
  (PREFILL→DECODE) 그대로.

### 2.2 per-iteration 배치 former — 핵심 (S2 구현 중 정밀화 2026-06-01)

> **발견(코드 정독 후):** persistent-mb "병" = mb 컨테이너 자체가 아니라 **(1) `_try_join`
> 유휴율 게이트**(합류 차단 → 슬롯이 "초기 전부 prefill → 후기 전부 decode" 코호트로 굳음)
> + **(2) per-mb KV 예산 "땜질" 프레이밍**. (나) disjoint 결정에서 **mb = 슬롯**, per-mb
> KV = *원리적 2-슬롯 분할*. → S2 = 유휴 게이트 제거(연속 backfill) + 재프레이밍.
> mb 컨테이너 전면 폐기 아님 (substrate 최대 재사용 — 프롬프트 "substrate 재사용·맹신 금지").

- [x] **재충전 = backfill 삭제 (S2 최종, 사용자 확정 2026-06-01).** 초기엔 "sticky 슬롯 +
  연속 backfill"로 봤으나, 동작점(세 시간 형성 시 균형)이 고정이라 **"균형 맞추려 이미 형성된
  배치에 더 합류"시킬 이유가 사라짐.** former 가 한 번에 동작점까지 형성 → 멤버는 자기
  μ-batch 안에서 단계 전이(prefill→decode)하며 돌다 완료 시 빠짐(슬롯 자연 축소). 새 부하는
  새 μ-batch(ADMISSION_TICK former)로만 진입 → **풀→배치 진입 경로가 former 하나로 단일화.**
  `_recompose_mb` = 잔존 멤버 phase 전진만(풀 pull 없음). `_backfill_slot` 삭제.
- [x] **disjoint 보장:** 진입이 former(request_queue pop = 한 번만 admit) 하나뿐 → 슬롯 간
  자동 disjoint. 요청 동시 1슬롯 불변식 성립.

- [ ] **밸런스 = KV 총량 동작점 (§0.8 확정).** prefill 을 동적으로 키우는 게 아니라:
  - **디코더를 풀에서 골라 담아 KV 합을 21.5M~29M(목표 25M) 범위에 넣음** = PIM 시간을
    B 시간(~101µs)에 맞춤. 디코드는 쪼개기 불가 → 범위로 수렴(사용자).
  - **prefill = 512 토큰 고정** (배치당, 2^9). 동적 chunk 사이징(`balance_pim_slack`)·
    유휴율 게이트(`balance_intra_A`) 불필요 — 동작점이 곧 답.
  - 슬롯당 KV 예산 = `_per_mb_kv_budget`(60M/2=30M, §0.8 A안)은 disjoint 2-슬롯 분할용으로 유지.
  - `balance_inter_AB`·`balance_pim_slack` **둘 다 삭제**(§2.5) — 동작점 고정이라 동적
    조정 불필요. 워크로드가 동작점 벗어나면 그건 레버로 못 고침(ctx 입력, §0.8) = 정상.
- [ ] **★ 생애 사이클(prefill→decode 전이)은 *반드시 유지* — 동적 밸런스 삭제와 무관.**
  동적 prefill/chunk 사이징(`balance_pim_slack` 등)은 없애지만, **요청의 생애 전이는
  풀 모델의 본질이라 남는다**: 트레이스의 한 요청은 prefill 을 거쳐 decode 로 가고,
  **prefill 이 끝나면(`prefill_processed ≥ len(prompt)`) 그 요청을 decode-only 로 전환**
  (`RequestState.PREFILL→DECODE`) → 다음 iteration 부터 PIM(decode-attn) 을 채운다.
  코드: [main_loop.py:345·353] `req.transition_to(DECODE)` — **S2 에서 유지(삭제 아님)**.
  → "밸런스 계산은 정적 동작점으로 대체, 요청 생애는 그대로" (사용자 확인 2026-06-01).
- [ ] **iteration 단위 = 1 forward pass(L=80 layer → decoder 당 1 token).** mb 의 layer
  반복(reset_micro_batch per layer)은 substrate 그대로. 바뀌는 건 L 도달 후 *재선택*.

### 2.2.1 tick — 고정 주기 polling 불필요 (이미 event-driven), 재충전 트리거는 유지

> 사용자 질문: "처음 배치 짤 때 정해지니 tick 도 필요 없지?" — 부분 맞음. 세 가지 분리:

- [x] **고정 주기 polling tick = 이미 없음.** STEP 2.5(Phase-1)에서 폐기됨
  ([main_loop.py:186] 주석 "고정 타이머 self-push 폐기"). 지금도 주기적으로 "비었나"
  계속 체크 안 함 — ADMISSION_TICK 은 **완료(KERNEL_COMPLETION)·신규 도착(REQUEST_ARRIVAL)
  이벤트에만** 재기동(line 129·137·144). `tick_interval_us`(10µs)는 그 재기동 지연값일 뿐
  주기 polling 아님.
- [ ] **매 tick cycle 측정·payload 계산 = 삭제(§2.5).** 동작점 고정이라 a_cycle/b_cycle/
  t_pim 측정 불필요 → ADMISSION_TICK payload 가 trivial 화.
- [ ] **빈 슬롯 재충전 트리거 = 유지(필수).** "한 번 짜고 끝"이 아님 — 요청이 완료돼 슬롯이
  비면 그 자리를 풀에서 다시 채워야 풀이 계속 돈다(sticky 슬롯 + backfill, §2.2). 이
  트리거(완료 이벤트 → former 재호출)는 풀 모델의 심장. 즉 ADMISSION_TICK *이벤트*는
  남되 payload 가 가벼워지고, trigger 는 이미 이벤트 기반.

### 2.3 staggering — μ-batch *다수의 존재 이유* = F3(일차) + F2(이차)

> **μ-batch 를 여럿 두는 *근본 이유* = F3 inter-instance 파이프라인** (사용자 강조
> 2026-06-01): 인스턴스 A(attention/proj) → B(FFN) → A(다음 layer) 가 μ-batch 1개면
> 직렬 → **B 가 FFN 도는 동안 A 가 논다.** 여러 μ-batch 면 B 가 M 의 FFN 하는 동안 A 가
> N 의 attention 을 돌려 A 를 안 쉬게 함 (ARCH §3.4·§5.7 F3, "PB1 eliminated",
> `t_A+t_B → max(t_A,t_B)`). 파이프라인 3단(A-proj / A-attn(PIM) / B-FFN) → window=3.
> F2(§5.6, A 내부 GPU proj ∥ PIM attn)는 그 위에 *추가로* 얹히는 이차 overlap.

- [x] **더블 버퍼링 (intra-A):** DAG 에 mb 간 edge 없음 → PIM 이 M decode-attn 하는 동안
  GPU 가 M+1 QKV backfill (dispatcher.refresh_ready + pick_gpu). 기판 제공, 별도 코드 불필요.
- [x] **F3 (inter-AB) — S0 에서 해결.** B-FFN 을 스케줄 노드(`NodeType.FFN`)+자원
  (`INSTANCE_B`)로 모델링(§2.7, commit 7c00532). `O_PROJ→FFN` 종속 + DAG mb간 edge 없음 →
  `FFN(M)` 이 INSTANCE_B 점유 중 GPU 가 `QKV(N)` backfill (test_phase2_ffn_stage 의
  test_f3_overlap 통과). 옛 telemetry-only 공백 해소.
- [ ] window(capacity=3) = 2 active + 1 전이 여유. former 가 활성 슬롯을 *명시적으로*
  채워 F3·더블버퍼링 발현하도록(S3 — 현재는 순차 충전이라 staggering 이 우연 의존).

### 2.4 admission control (동시 decoder 한도)

- [ ] decode-set 크기 > cap 이면 신규 prefill admit 정지(큐잉) — 표준 throughput/latency
  trade-off. KV 캐파 초과도 동일. 현 `layer1` head-of-line walk(109~145) 로직 재사용.
- [ ] **TTFT↔TBT 정책:** 슬랙 0(GPU 이미 PIM 보다 김)일 때 최소 prefill 보장 여부 —
  기본 *무보장*(TBT 우선, 배치_생애 §밸런스). 옵션으로 floor 노출만.

### 2.5 ★ 동작점 고정이 코드를 단순화한다 (사용자 통찰 2026-06-01)

> §0.8 에서 밸런스가 **"KV 합을 21.5~29M 범위에 들도록 디코더 골라 담기 + prefill 512
> 고정"** 으로 확정 → **매 tick 동적 측정·계산하던 기계장치가 통째로 moot.** former 가
> "측정→cycle 비교→chunk 사이징"이 아니라 "KV 합 채우기 + prefill 512"로 바뀜.

**moot 되는 것 (S2 에서 삭제·격하):**
- [ ] `_compose_admission_payload`(207~263) — a_cycle/b_cycle/t_proj/t_pim_fn/per_token
  5개 payload 산출. **대부분 불필요.** 동작점이 고정이라 매 tick 측정 안 함.
- [ ] `_measure_cycles`(272~285) + `_prev_a/b_active_snapshot` — cycle delta 측정. moot.
- [ ] `_make_t_pim_fn`(287~305) — t_pim closure. former 가 KV 합 직접 보면 됨, closure 불필요.
- [ ] `admission.balance_inter_AB`·`balance_pim_slack` — 동적 prefill 사이징. prefill 512
  고정이면 둘 다 불필요(REPORT: 512 만 균형). deadband(`deadband.py`)도 같이 moot.
- [ ] `admission.layer1` 의 t_proj·t_pim_fn·a_cycle·b_cycle·gpu_op_time_per_token 인자 —
  전부 제거. layer1 = "KV 합 범위까지 디코더 admit + prefill 512" 로 축약.
- [ ] `_fire_admission_tick`·`AdmissionSnapshot` 의 cycle 필드 — 측정 진단용으로만 격하
  (evaluator 가 idle_telemetry 로 사후 산출, admission 경로에서 분리).

**남는 핵심 (단순):**
- former: 풀에서 디코더 골라 **Σkv ∈ [21.5M, 29M]** + **prefill 512** → 배치. 끝.
- KV 캐파 = 배치당 30M·총 60M aggregate = OOM 천장(§0.8 A안). window 3 = disjoint 2슬롯+여유.
  dispatcher/DAG/PIM/FFN substrate 그대로.

> **풀 구성 (warm-start, §2.6):** 정상상태 풀엔 (a) decode-only 다수(prefill 끝남) +
> (b) prefill-중 소수 + (c) 그 prefill 에 종속된 decode. former 는 (a)+(c) 로 KV 합을
> 채우고 (b) 에 prefill 512 배분. 트레이스 = 한 요청 prefill→decode 종속 그대로.

**예상 효과:** main_loop 에서 측정·payload·closure ~100 LOC 감소. admission 은 balance
3종 메서드 삭제로 절반. 사용자 직관("생각보다 단순해진다") 맞음 — 단 *측정 substrate*
(idle_telemetry·evaluator)는 **진단용으로 보존**(밸런스 입력에서만 분리).

### 2.6 측정용 warm-start seed (채택 = B)

- [ ] measure 하네스에 "사전 decode 풀 seed" init 추가: t=0 에 *이미 decode 중인 요청들*
  (kv_length·남은 decode 분포)을 in_flight_requests + DECODE 상태로 주입 + KV admit.
- [ ] 트레이스 포맷 불변(`arrival_time,prompt_len,max_tokens`). seed 는 *초기상태 확장*이지
  워크로드 표현 변경 아님.
- [ ] **caveat:** seed 분포가 현실 정상상태(ctx·decode 진행도)를 대표해야 편향 없음.
  분포는 워크로드에서 유도 또는 cold-start 1회 관측치로 seed.

### 2.7 F3 / Instance B 모델링 — B-FFN 을 스케줄에 반영 (확정 = 방식 i, S0 완료)

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

- [x] **S0. F3 / Instance B 스케줄 모델링** (§2.7 방식 i) — 완료. `NodeType.FFN`(node.py),
  `O_PROJ→FFN` edge(dag.py), `INSTANCE_B` 자원 + `pick_instance_b`/`dispatch_instance_b`/
  I6(dispatcher.py), layer advance 트리거 O_PROJ→FFN(main_loop.py), `check_I6`(invariants.py).
  신규 테스트 `tests/test_phase2_ffn_stage.py` 9개 통과(DAG 5노드·FFN ready=O_PROJ done·
  dispatch·I6·단일 layer FFN 종료·F3 overlap[FFN(M) B점유 중 GPU 가 QKV(N) backfill]).
  미정리: 옛 `instance_pipeline.dispatch` telemetry 경로(이제 FFN 노드가 대체) + 4노드
  가정 옛 테스트(test_dag/test_dispatcher/test_invariants 등) — S2 후 일괄 갱신.
- [x] **S1. `admission.py` 슬림화** — 완료. `balance_intra_A`(유일 유휴율 레버) + layer1
  호출 삭제. 테스트 정리: test_admission intra_A 4개·test_chunk_size intra-A 2개·
  test_idle_telemetry chain 1개 제거, idle_telemetry.py 주석 갱신. **84 tests green**.
  ※ S1 당시엔 `balance_inter_AB`·`balance_pim_slack` 유지로 봤으나, **이후 §0.8 동작점
  확정으로 둘 다 S2 삭제 대상으로 변경**(§2.5). `max_mb_kv_tokens` 는 S2 에서 처리.
- [x] **S2. 동작점 former + 측정/밸런스/backfill 기계장치 제거** (§2.5) — 완료.
  - `admission.layer1` 동작점 former 재작성: `balance_inter_AB`·`balance_pim_slack`·`mfu_floor`
    삭제, t_proj·t_pim_fn·a_cycle·b_cycle·per_token 인자 제거. 종료 = Σkv ≥ 25M(목표) + prefill
    512 고정 + per-mb·전역 KV. `MicroBatchSpec.n` 삭제(N_dec=len 과 중복).
  - `main_loop`: `_compose_admission_payload`·`_measure_cycles`·`_make_t_pim_fn`·`_last_dispatched_mb`·
    `_prev_a/b_active_snapshot` 삭제. ADMISSION_TICK payload trivial(빈 dict). `_fire_admission_tick`
    cycle 인자 제거(snapshot a/b_cycle=0 — 진단은 idle_telemetry 로 분리).
  - **backfill 삭제**(사용자 확정): `_backfill_slot` 제거, `_recompose_mb`=잔존 멤버 phase 전진만.
  - `max_batch_size` config 필드 제거(N_dec 부산물). `_per_mb_kv_budget`=비바인딩 선언 유지(각주).
  - config: `kv_capacity_aggregate` 4M→60M, `kv_operating_target_tokens`=25M 추가.
  - L 도달 token 생성·생애 전이(prefill→decode) 유지 확인.
  - 테스트: test_admission 재작성(21 green) + balance/payload/backfill 테스트 4파일 폐기 +
    test_admission_tick·test_meta 갱신(직접 영향 29+13 green). **사전-깨짐(S0 O_PROJ→FFN
    트리거) 34건은 HEAD 에서도 red(stash baseline 확인) → S3 일괄 갱신.**
- [~] **S3. 사전-깨짐 테스트 마이그레이션 + run.py 회귀 수정** — 대부분 완료.
  - **★ run.py 회귀 수정**: S2 가 `_compose_admission_payload` 삭제했으나 [run.py:164] 가 호출 →
    `Run.init` 깨짐(Run 기반 테스트 전부 fail). payload={} 로 수정. (S2 때 src 호출처 grep 누락 — 교훈.)
  - O_PROJ→FFN 트리거 마이그레이션: test_main_loop_completion(25 green, 의미 테스트 2개 FFN 기준
    재작성), test_cross_module_lifecycle(`_decode_one_token` FFN; 16/17 green).
  - 4노드→5노드: test_meta(node_types·dag_precedence +FFN), test_dag(precedence +FFN, 4→5 rename).
  - ns/µs: test_cross_module_pipeline(_op_time µs 기대값 ×1e-3).
  - telemetry: test_instance_pipeline_dispatch_invoked_per_layer 폐기(dispatch hot path 제거됨).
  - acceptance c1·c2·c4·c5·f4(25) + inter_ab(7) green. **잔여: 실트레이스(longbench/e2e) OOM** —
    run.py eager-preload 이 `prompt_tokens=[0]*num_prefill` 로 ~1M-ctx 통째 materialize(스케줄러는
    len 만 씀). harness 낭비 → **S4 에서 Request 를 prompt_len 으로 경량화** 필요.
  - window/F2 명시 2슬롯 유지: 코드 추가 보류 — event-driven admission 의 emergent staggering 을
    S4 측정으로 먼저 확인(idle≈이론), 미발현 시 그때 보강(§7 추측 금지).
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
  - former: 디코더 골라 Σkv 가 [21.5M,29M] 범위 도달 시 정지 / prefill 512 고정 /
    cap·KV캐파(배치당 30M / 총 60M) 초과 시 큐잉. (cycle 측정 없음 — 동작점 직접.)
  - 전이: prefill 소진 → DECODE → 다음 iteration decode-set 포함.
  - FFN(S0): O_PROJ→FFN 종속, layer 경계=FFN 완료, F3 overlap.
- [ ] **메타-테스트(플랜 정합):** 삭제 대상 심볼(`_try_join`·`balance_intra_A`·
  `balance_inter_AB`·`balance_pim_slack`·`_measure_cycles`·`_make_t_pim_fn`) 부재 단언 /
  NodeType(5: +FFN)·RequestState enum 완전성 / DAG edge=I1~I3+O_PROJ→FFN.
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
- [ ] **warm-start seed init**(§2.6) 하네스 플래그 추가(`--seed-decode-pool`).
- [ ] **트레이스 = 기존 generator 재사용**(포맷 불변):
  - `gen_step5_traces.py` — ts(저ctx 128~1K)·tgen(중ctx 2~8K)·agentic(고ctx 32~128K).
  - 실행(§9 습관 준수, 백그라운드 하나씩):
    `PYTHONIOENCODING=utf-8 python debug_phase2/measure_steady.py --trace data/agentic_30.csv
    --seed-decode-pool ... > debug_phase2/out_agentic.txt`
- [ ] **★ idle ↔ spread 성공 기준 (정량, 사용자 2026-06-01).** 완벽 overlap 이면 cycle =
  max(PIM, GPU-A, B), 각 자원 idle = (max − busy)/max. 따라서 **세 시간 spread = 가장
  한가한 자원의 idle**. 검증 기준:
  - **완전 균형(동작점 KV 25M)**: 세 자원 idle **≈ 0%** (이론 spread 0.6%). 의도한 최적점
    — 아무도 안 기다림 → TTFT·TBT·throughput 동시 최적.
  - **허용 범위(KV 21.5~29M)**: 가장 한가한 자원 idle **≤ 15%** (하한선 PIM·GPU idle ~13%,
    상한선 B idle ~14%). 워크로드 변동 흡수 폭.
  - **이론 ↔ 실측 일치 여부 = Phase-2 핵심 성공 판정.** 위는 *완벽 overlap 가정*. 실제론
    같은 μ-batch 내 QKV→attn 직렬이라 100% overlap 아님 → F2/F3 staggering 이 연속 μ-batch
    겹쳐 근접시킴. **실측 idle 이 이론(≈0~15%)에 얼마나 가까운지**가 F2/F3 발현의 증거.
- [ ] **검증 기대치 (스케일 스펙트럼):**
  - 고ctx(agentic, 요청 ctx ~87K~117K): 동작점 안 → 세 자원 idle ≈0~15%, TBT≈cycle 바운드
    유지(STEP5 TBT 폭증 재현 안 됨이 성공 신호).
  - 저ctx(<7K, TP 픽스 후 임계): PIM < GPU 라 PIM 유휴 *정상*(물리, ARCH §6.6). 풀의 KV 합이
    21.5M 에 못 미치면 동작점 former 가 작은 배치로 흡수 → FFN 작아 빨리 끝남(§0.9). PIM
    유휴는 측정·기대(고칠 대상 아님).
  - cross-check: op-time 직접 산출(diag) = 하네스 t_pim/t_gpu/t_b 일치 확인.

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
- [ ] **검증은 철저하되 과도하지 말 것** (사용자 2026-06-01). 변경한 모듈+직접 영향
  테스트는 *철저히* 돌려 확인하되, 무관한 전체 스위트를 반복 회귀하거나 같은 걸 여러 번
  돌리지 않는다. 메타·음성·교차불변 테스트도 *핵심만* — 과잉 케이스 양산 금지(CLAUDE.md
  §2 단순성). "green ≠ correct" 는 지키되 테스트를 위한 테스트는 만들지 않는다.

---

## 알려진 사전-깨짐 테스트 (TP 픽스 무관, 별도 정리 필요)

- [ ] `test_cross_module_inter_ab_wiring::test_instance_pipeline_dispatch_invoked_per_layer`
  — S0 가 `instance_pipeline.dispatch` 를 hot path 에서 제거(FFN 노드 대체) → 호출 0.
  의도된 결과. 테스트 폐기 또는 FFN 노드 기준으로 재작성 (S3).
- [ ] `test_cross_module_pipeline::test_admission_to_dispatch_pim_op_time_chain` +
  `::test_multiple_micro_batches_independent_signal_flow` — PIM op_time 을 ns(267.5)로
  기대하나 dispatcher 는 µs(0.2675) 반환. 사전부터 깨짐(S0/S2 전). 단위 기대값 수정 (S3).

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
  미리"). 방식 (i) 단일 이벤트루프 + INSTANCE_B 자원(§2.7). 재설계 범위가 기판(node·dag·
  dispatcher)까지 확장. **구현 순서 변경: S0(F3 모델링) → S1(admission) → S2(former) → …**
  S0 가 former 의 토대라 먼저. 다음: S0 착수.
- 2026-06-01: **★ TP=8 분산 버그 발견·수정**(사용자 지적, commit 40e812a). GPU/FFN 이
  단일 GPU peak 으로만 나눠 8배 과대. PIM=GPU 임계 56K→7K, 삼중균형 280K→100K(§0.7).
  모든 균형 결론의 토대 정정. 사전-깨짐 3개 식별(별도 정리).
- 2026-06-01: **★ 동작점 KV 총량 기준 확정(§0.8).** 스케줄러 레버 = 디코더 골라 KV 합
  맞추기(쪼개기 불가 → 범위 수렴). 목표 KV 25M, 허용 21.5~29M(15% 오차), prefill 512 고정,
  균형 ~101µs. **KV 캐파 = 30M 확정**(넉넉). 타깃 = 요청 ctx ~87K~117K long-context agentic.
  vLLM 512 는 토큰(K 아님, web 확인) — 경험적 타협점 vs PULS 는 하드웨어 균형서 유도.
- 2026-06-01: prefill 1024/2048 sweep — **512 만 균형, 1024+ 는 GPU-A(PREFILL_ATTN) 폭주로
  불가**(REPORT.md, commit 0e1f200).
- 2026-06-01: **동작점 고정 → 동적 밸런스 기계장치 moot(§2.5)**(사용자 통찰). former =
  "KV 합 채우기+prefill 512"라 `_compose_admission_payload`·`_measure_cycles`·`_make_t_pim_fn`·
  balance 3종·deadband 삭제 대상. main_loop ~100 LOC↓, admission 절반↓. 측정 substrate
  (idle_telemetry·evaluator)는 진단용 보존. S2 재정의(§2.5·§3 S2). 다음: S2 former 구현.
- 2026-06-01: **생애 전이 유지 + tick 구분 명확화(§2.2.1)**(사용자). admission 호출은 당연히
  필요(빈 슬롯 재충전) — 없애는 건 *주기적 체크 tick*뿐(이미 STEP 2.5 에서 폐기). 생애
  전이(prefill→decode)는 동적 밸런스 삭제와 무관하게 유지(§2.2 ★).
- 2026-06-01: 가속 커버리지 확인(§0.6) — 동역학 2(더블버퍼링·인스턴스A∥B), op-time 2
  (F1·F5), closed-form 2(Aux). F2→"더블 버퍼링" 용어 통일. **분할 작업: 1차 S0~S2,
  2차 S3~S5** (사용자). 옛 전체 회귀 안 돎(다 교체) — 변경별 타깃 테스트만. test_invariants.py
  는 깨진 placeholder(마크다운 쓰레기 혼입) — 어차피 재작성.
- 2026-06-01: **동작점 재실측 재현 + 캐파 의미 확정(§0.8 A안)**(사용자). op-time 직접 산출로
  삼중 균형 재현: KV 25M/배치 → PIM 102/GPU-A 101/B 101µs, spread **0.7%**; 밴드 21.5M→12.4%,
  29M→13.4% (≤15%). **25M = 마이크로 배치 *하나*의 decode-attn op KV (PIM op_time 1회 호출)** —
  슬롯 분할 아님(이전 보고의 12.5M 오독 정정). 캐파 = **배치당 30M, 총 60M aggregate**(A∥B
  오버랩 위해 2 배치 동시 in-flight, disjoint·영구 상주). HW 가정 80GB→**160GB/stack(10.24TB,
  61.7M 담음)** 2배. `_per_mb_kv_budget = 60M/2 = 30M` 자동 정합 → 이전 "cap/2 가 동작점 막음"
  우려 해소. config `kv_capacity_aggregate` 4M→**60M**(S2/S5). README HW 스펙도 갱신 대상(§6).
- 2026-06-01: **S2 구현 완료(동작점 former + 측정/밸런스/backfill/max_batch/mfu 제거).** 검증
  대화 중 추가 단순화 확정(사용자): (1) **max_batch_size 제거** — N_dec 은 부산물(=25M÷ctx),
  개수 캡은 동작점이 먼저 멈춰 안 걸리는 중복 레버. ctx 짧/길 모두 cycle≈102µs 로 묶임을
  op-time 산출로 재확인(짧으면 PIM 유휴·정상). (2) **backfill 제거** — 세 시간이 형성 시점에
  맞춰져 "균형 맞추려 더 합류" 이유 소멸. former 단일 진입. (3) **mfu_floor/MicroBatchSpec.n
  제거**(사문/중복). (4) per-mb 예산은 비바인딩 선언으로 보존(각주). 디코드→QKV/O-proj 기여가
  balance 산식에 포함됨도 확인. 사전-깨짐 34건(S0 O_PROJ→FFN 트리거)은 baseline red → S3.
- 2026-06-01: **S3 테스트 마이그레이션 + run.py 회귀 수정.** ★ S2 가 `_compose_admission_payload`
  삭제 시 [run.py:164] 호출처를 놓쳐 `Run.init` 깨짐 → payload={} 로 수정(Run 기반 전부 복구).
  O_PROJ→FFN 트리거 마이그레이션(완료/lifecycle/dag/meta), ns/µs 기대값, dispatch_invoked 폐기.
  검증: acceptance c1·c2·c3·c4·c5·f4 + e2e + inter_ab = **39 pass**(+retired 1), lifecycle 16/17,
  main_loop_completion 25, meta 53, admission 21 — 전부 green. **유일 잔여 red = test_real_longbench
  (실 1M-ctx 트레이스 eager-preload 의 `[0]*num_prefill` materialize → OOM, harness 낭비). 스케줄러
  는 len 만 쓰므로 S4 에서 Request prompt_len 경량화로 해소.** F2/F3 명시 2슬롯은 S4 측정 후 판단.
  - **교훈(자기리뷰 보강): 심볼 삭제 시 tests 뿐 아니라 src 호출처도 grep** (run.py 누락 재발 방지).
