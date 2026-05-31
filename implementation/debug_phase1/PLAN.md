# Phase-1 Debugging Plan — Balance 4-Factor 발현 검증

## 1. 배경 및 계기

현 스케줄러는 실 트레이스(LongBench λ=3.40)에서 PIM Instance A idle 99.66%를
기록한다. 본 수치는 두 원인의 곱으로 분해된다.

- **트레이스 성질** — decode 토큰 수가 전 요청 350 고정(분산 0), prefill 평균
  322,882 대비 약 920× 작다. PIM이 처리할 decode-attention 일감이 구조적으로 희소.
- **Balance 4-factor 미발현** — intra-A 균형(PIM busy → GPU에 chunked prefill,
  GPU busy → PIM에 chunked decode)이 admission 레벨에서 실효 없이 구현되어 있다.
  - `balance_pim_slack` (chunked prefill) — 산식은 존재하나, 끼울 prefill은 *해당
    mb 내부의 잔여 prefill 요청*으로 한정된다. 진행 중 배치에 신규 요청을 합류시키는
    경로가 부재하여, prefill 소진 후 mb는 순수 decode로 전락한다.
  - `balance_intra_A` (chunked decode) — 반환 `decode_count+1`이 `spec.n`(telemetry
    snapshot)에만 반영되고 실제 배치 구성에 미반영. 실효 없음.

분석 결과 순수 decode 배치의 PIM/GPU 시간비는 요청 수 N에 무관하며 요청당 평균
컨텍스트로만 결정된다 (PIM/GPU = ctx / 56,160). 따라서:

- **ctx > 56K (long-context)** — PIM 바운드. idle 불가. 이때 GPU가 유휴이며,
  chunked **prefill** 합류로 GPU 빈자리를 메워야 한다.
- **ctx < 56K (short-context)** — 요청 추가 시 GPU projection이 더 빠르게 증가하고
  짧은 컨텍스트는 단일 PIM 타일(65,536 row)에 packing되어 PIM이 거의 증가하지 않는다.
  PIM idle은 **정상 동작**이며 balance 수정 대상이 아니다.

## 2. 목적

balance 4-factor 미발현을 합성 트레이스로 확증하고, 합류 경로 신설 후 idle 감소를
인과적으로 입증한다.

## 3. 합성 트레이스 설계

세 트레이스는 각기 다른 명제를 검증한다. 공통적으로 decode 토큰 수 분산을 높이고
(현 350 고정 탈피) 도착률을 포화시킨다 (idle 분모 = wall-clock span 오염 방지).

| ID | regime | ctx 대역 | 기대 동작 | 검증 명제 |
|---|---|---|---|---|
| **T-S** | short 몰림 | ≪ 56K | PIM idle 큼 (정상) | 음성 대조군 — 수정 후에도 idle 유지되어야 정상 |
| **T-L** | long 몰림 | ≫ 56K | PIM 바운드, GPU 유휴 | chunked prefill 합류 버그 노출 |
| **T-M** | 실제형 혼합 | short+long (≈3:7) | 시점별 바인딩 반전 | 양방향 balance 동시 작동 필요 |

제약 (코드 근거):

- KV capacity 4,000,000 (`config.py`) — 동시 admit 요청의 kv_length 합 상한.
  T-L은 ctx 과대 시 동시성이 1~2로 붕괴하므로 56K–150K 대역에서 동시 수십 개 생존
  하도록 조절.
- decode = max_tokens (`completion.py`) — decode 분산이 곧 decode 단계 체류 시간
  분산. 로그정규/지수 계열로 부여.

생성 방식: `synthesize()` 미수정. 별도 스크립트로 CSV 3종을 `debug_phase1/data/`에
산출하고 `Run`에 path로 주입 (소스 무수정, 재현성 확보).

## 4. 실행 절차

0. **Baseline 고정** — 3 트레이스를 현 코드로 실행, `idle_fraction` 3-key +
   dispatch_trace 수치 저장 (수정 전 "before").
1. balance 4-factor 미발현 확증 — T-L에서 PIM 바운드·GPU 유휴 구간 발생, 신규 prefill
   미합류를 dispatch_trace로 확인. `balance_intra_A`의 `+1` 무실효를 단위 수준 확인.
2. 합류 경로 신설 — 진행 중 배치에 신규 요청 prefill/decode 합류 (양방향). window
   capacity 한도 동반 검토.
3. 재실행 — 동일 트레이스·도착률·span 조건 유지. baseline 대비 idle 변화 측정.

## 5. 판정 기준

- **주 지표** — `idle_fraction["pim_instance_a"]` (record_active 기반). `pim_utilization`
  은 dispatch 간격 적산·마지막 completion 미반영으로 신뢰하지 않는다.
- **인과 분리** — 도착률·span을 수정 전후 동일하게 고정. idle 변화가 balance 수정에만
  귀속되도록 통제.
- **합격 조건**
  - T-L — 수정 후 GPU(또는 종합) idle 유의 감소.
  - T-M — 양방향 발현으로 PIM·GPU idle 동시 최소화.
  - T-S — idle 실질 불변 (과잉 수정의 음성 대조).

## 6. 체크리스트

### 합성 — 완료
- [x] CSV 생성 스크립트 작성 (`generate_traces.py`)
- [x] T-S 생성 — short 몰림(ctx>56K 0%), decode 고분산(97–1226), 도착 포화
- [x] T-L 생성 — long 몰림(ctx>56K 100%), 58K–150K, 동시 admit ~37.6
- [x] T-M 생성 — short+long ≈3:7(ctx>56K 72%), prefill heavy-tail 755K
- [x] 분포 sanity (ctx/decode min·mean·max, 동시 admit 추정)
- [x] 캐파 압박 트레이스 추가 (`gen_long_pressure.py` — 누적 KV 7.88M > 4M)

### Baseline (수정 전) — 완료
- [x] 축소 T-L 실행, idle 3-key + dispatch_trace 저장 (`REPORT_baseline.md` §2–6)
- [x] 근본 원인 정밀 특정 — 단일 mb 귀결 (`trace_single_mb.py`, §5–6)
- [x] 캐파 압박에도 합류 부재 확증 (`prove_no_join.py`, §7 — join=False)
- [x] 직렬 mb 처리 확증 (`prove_serial_fast.py`, §7b — window 내내 1)
- [x] (정정) 근본 원인 = "한 tick=캐파까지 한 mb" → 직렬 처리. balance·staggering·
      합류 발현 무대 자체 부재 (단일 원인)

> 순서: **1차 수정 → 1차 회귀 → 2차 수정 → 2차 회귀 → 재검증**.
> 1차에서 mb 다중화가 먼저 돼야 2차 합류가 의미를 가지므로 단계 분리.

### STEP 1 — 수정 1차: 세 한계 분리 (동시 다중 mb 형성의 토대)
> 코드 정독 결과 범위 조정: 직렬 처리의 직접 원인은 release/evict 가 아니라 **새 mb
> 미생성**(첫 tick 에 가용 요청을 캐파까지 한 mb 로 다 admit → 큐 즉시 빔 → 이후
> `layer1` 이 admit 대상 0 → `return None`). 따라서 핵심은 **seq 상한 하나**.
> - 요청별 즉시 KV release 는 **이미 구현됨** (completion.finalize → kv.release,
>   main_loop §351–353). STEP 1 에서 손댈 것 없음.
> - evict 조건 변경("전원 완료 AND 큐 빔")은 **STEP 3 으로 이동** — 합류가 있어야
>   "큐 있는 한 mb 유지"가 의미. STEP 1(합류 없음)에서 바꾸면 evict 만 지연되어
>   캐파가 묶임.

- [x] **배치 크기(seq 상한) 신설** — config `max_batch_size`(기본 256). 요청 개수
- [x] 실제 배치 크기 = min(seq 상한, KV 캐파 허용분) — admission.layer1 루프에 적용
- [x] **window = 3 고정 명시** — window.py 주석 보강 (DEFAULT_CAPACITY=3, 스윕 X)

### STEP 2 — 1차 회귀 (타깃 범위만) — 완료
- [x] 타깃 모듈 테스트 — **269 passed, 0 failed**
- [x] 깨진 테스트 2건 업데이트: test_admission_tick_rescheduling(payload 6키, 사전버그)
      + test_meta(_EXPECTED_ADMISSION_FIELDS 에 max_batch_size 추가)
- [x] mb 다중화 인과 분리 검증 (`prove_multiplex.py` → `multiplex_result.txt`):
      동일 트레이스·캐파, seq 상한만 다르게 → A(무제한)=직렬 window 1 /
      B(seq 2)=다중 window 2. STEP 1 효과 확증.

### STEP 3 — 수정 2차: 합류 경로 (양방향)
- [ ] 신규 요청의 in-flight mb 합류 경로 신설 — admission 이 request_queue +
      in_flight 둘 다 보도록. 합류 가능량 = min(seq 여유, KV 여유)
- [ ] prefill 합류 (chunk 로 잘라서) + decode 합류 (통째로)
- [ ] **합류 게이트 = 밸런스 in_band 조건** — 무조건 합류 아님. PIM/GPU 레이턴시
      맞으면(in_band) 합류 중단 (배치_생애 §5 인과: 밸런스→게이트 닫힘→소진→종료)
- [ ] **mb evict 조건 변경** — "전원 완료" → "전원 완료 AND 큐 빔" (STEP 1 에서 이동).
      합류가 생겼으므로 큐 있는 한 mb 유지 (배치_생애 §5 종료 조건)
- [ ] chunked decode 실효화 (`balance_intra_A` +1 이 실제 배치 반영되도록)

### STEP 4 — 2차 회귀 (타깃 범위만)
- [ ] 동일 타깃 모듈 테스트 재실행 (`test_admission/main_loop/completion/window/...`)
- [ ] 깨진 테스트 업데이트/수정

### STEP 5 — 재검증 (수정 후) — 새 검증 테스트 중심
- [ ] 배치 크기 스윕 {256, 512} — 각각 idle_fraction + (대리)TTFT/TBT 관측
- [ ] token budget closed-form 산출 (트레이스 decode/prefill 기반, 모델 미실행)
- [ ] T-L — GPU/종합 idle 감소 확인
- [ ] T-M — 양방향 idle 동시 감소 확인
- [ ] T-S — idle 불변 확인 (대조군, 과잉 수정 방지)
- [ ] before/after 보고서 작성

> 미수정 영역(RTL·evaluator·trace 생성 등)은 회귀 스킵.

## 7. 산출물

- `debug_phase1/data/` — 합성 CSV (T-S/T-L/T-M + 압박 + 직렬검증 tiny)
- `debug_phase1/*.py` — 생성·계측 스크립트 (generate_traces, gen_long_pressure,
  analyze, trace_single_mb, prove_no_join, prove_serial_fast, config_small_cap)
- `debug_phase1/REPORT_baseline.md` — baseline 관측·근본 원인·수정 설계
- `debug_phase1/serial_result.txt` — 직렬 처리 검증 결과
