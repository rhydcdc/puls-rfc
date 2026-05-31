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

### STEP 2.5 — admission 구동 모델 수정 (이벤트 기반) ※ STEP 1 측정 중 발견
> 발견(REPORT §10): admission tick 이 고정 10µs 타이머로 self-reschedule 되어, GPU 가
> 긴 op(예 prefill chunk 8192) 도는 동안 admit 불가 상태에서 헛돈다 (KC 1건당 tick
> 수~수십). 라이브락 아님(진행은 함) 이나 step·메모리 폭증. balance·합류가 강해질수록
> chunk↑ → 더 자주 발생. STEP 3 전 선결 (합류도 admission 경로에 얹힘).

- [x] **KERNEL_COMPLETION 에 admission 시도 추가** — 완료 = iteration 경계 = admit 기회
- [x] **고정 타이머 self-push 제거** — `_schedule_next_admission_tick` 함수 삭제(orphan)
- [x] **REQUEST_ARRIVAL 트리거 유지** — cold start 재기동 보존
- [x] 재기동 단일 경로 = `_schedule_admission_tick_with_default_payload` (완료/도착 시)
- [x] 타깃 회귀 284 passed — TestSelfRescheduling 4건 이벤트 기반으로 업데이트
      (+ wiring 테스트 inspect 대상 함수명 갱신)
- [x] light_pressure 헛도는 tick 제거 확증: **3,000,000+ (not drained) → 76,820 (완주)**

### STEP 3-a — 합류 경로: prefill 방향 (PIM 바쁠 때 GPU 빈자리 채움) — 완료
- [x] 신규 요청의 in-flight mb 합류 경로 신설 — `_try_join_prefill` (`_recompose_mb` 내).
      request_queue 에서 들임. 합류 가능량 = min(seq 여유, KV 여유), FIFO
- [x] prefill 합류 (chunk 로 잘라서) — joined req = KV admit + in_flight + PREFILL 전이
- [x] **합류 게이트 = gpu_idle > idle_theta_high** (GPU 빈자리 있을 때만; PIM-bound 구간).
      GPU 포화면 게이트 닫힘 (배치_생애 §4·§5)
- [x] **mb evict 조건 변경** — "전원 완료" → recompose(자기재구성+합류) 후 prefill_chunk·
      decode_tokens 둘 다 비면 evict (배치_생애 §5: 완료 AND 합류 불가). 커밋 1080705
- [x] 단위 5 (test_prefill_join): 게이트 on/off, seq-bound, KV-bound, empty noop
- [x] 사전버그 7 (test_cross_module_lifecycle, 제 작업 무관) 수정 — `_drive_until_done`
      이 prefill_chunk 까지 보도록 + 직접구동 2건 prompt=[] isolate
- [x] 회귀 42 passed (lifecycle 17·stress·e2e·determinism·structural·f4). 푸시 1080705

### STEP 3-b — 합류 경로: decode 방향 (GPU 바쁠 때 PIM 빈자리 채움) — 완료
- [x] decode 합류 — `_try_join` 게이트 양방향 일반화 (gpu_idle>θ OR pim_idle>θ).
      decode 는 통째로(autoregressive), prompt 유무로 _populate_mb_phases 자동 분류
- [x] **chunked decode 실효화** — `balance_intra_A` 의 무실효 decode +1 제거.
      decode 조절은 _try_join 전담(실효 있게, 매 완료 경계). 시그니처 단순화
- [x] 단위 테스트 — test_prefill_join 에 decode 방향 3건 추가 (8건). 커밋 71abfe4

### STEP 4 — 3-b 회귀 (타깃 범위만) — 완료
- [x] 가벼운 타깃 회귀 283 passed (admission/main_loop/completion/window/config/meta/
      prefill_join/idle_telemetry 등)
- [x] 풀 회귀 49 passed (lifecycle 17 실트레이스 포함·stress·e2e·acceptance) — 커밋 게이트
- [x] balance_intra_A 시그니처 변경 영향 테스트 갱신 (test_admission·test_idle_telemetry)
- [x] 커밋·푸시 완료 (71abfe4 소스, 24af380 문서) — origin/main 동기화

### STEP 5 — 재검증 (수정 후) — 새 검증 테스트 중심
- [ ] 배치 크기 스윕 {256, 512} — 각각 idle_fraction + (대리)TTFT/TBT 관측
- [ ] **합류 게이트 임계(θ_high) 스윕 {0.3, 0.1}** — 목적이 "유휴율 0 수렴"이므로 30%는
      느슨(최대 30% 유휴 용인). 10%로 조이면 목적 충실하나 합류 잦아 진동 위험 → 실측
      비교. 게이트 기준 = 유휴율(idle_fraction), 레이턴시(in_band)와 동치이나 목적 지표라
      더 직접적 (배치_생애 §5).
- [x] **hysteresis 부활 (idle_theta_low)** — 완료 (commit 9368edf). θ_low 0.1→**0.05**
      확정 (θ_high 스윕 두 값 0.1·0.3 모두보다 작아 양쪽 유효 deadband). 신호 =
      max(gpu_idle, pim_idle), 닫힘→θ_high 초과 시 열고 열림→θ_low 미만 시 닫는 latch
      (`_join_gate_open`). **범위 = 합류 게이트(`_try_join`)만** (balance_intra_A 연속조절은
      무관). README 'adaptive admission with hysteresis deadband' 명시 기능 완성이라
      *측정 무관 채택* (진동 트레이스/순간신호 전환 대비 보험). 단위 13 passed
      (latch 전이 5 신규), 타깃 회귀 152 passed.
> **측정 프로토콜 결정 — 완주 안 함, 수렴 기반 조기 종료 (엄밀판).** 완주는 (a) 비용
> 과대 (T-L decode 2000×80층) + (b) 막판 도착 끊긴 ramp-down 꼬리가 idle 오염. 대신
> *워밍업(전 도착 주입 완료까지) → idle_telemetry.reset → 정상상태 윈도우에서 누적 idle
> 이 수렴(Δ<ε)하면 정지*. before/after·스윕에 **동일 프로토콜** 적용(delta 가 결론).
> 스크립트 = `debug_phase1/measure_steady.py`.
- [ ] (참고) pim_slack 안전마진(0.9 = t_pim×0.9, PIM 을 compute-bound 뒤 은닉)은 생존 중
      — balance_pim_slack 의 prefill chunk 산식. 이번 변경 무관, 유지.
- [ ] token budget closed-form 산출 (트레이스 decode/prefill 기반, 모델 미실행)
- [ ] **트레이스 재설계** — 구 T-L/long_pressure 는 prefill 지배라 (a) prefill 구간엔 decode
      일감 부재 → 합류가 PIM 못 채움, (b) 측정이 prefill 구간에 갇힘. 합류 효과의 핵심 =
      **decode-bound 정상상태에서 prefill backlog 로 GPU 를 채움**. 그에 맞춰 재설계:
  - **T-DEC (주 demonstrator)** — ctx 58–72K(임계 바로 위 → decode PIM-bound, prefill 은
    짧음), decode 200–800(고분산, prefill ~115cyc 지배 → run 이 decode-bound), N≈80(ΣKV>4M →
    prefill backlog 상존), 포화 도착. 합류 ON 시 backlog prefill 이 GPU 채워 **GPU idle 급감** 기대.
  - **T-M (현실성)** — 혼합 short+long · decode 분산. PULS 타깃(agentic·멀티턴·KV 길이 분산)에
    가장 근접. 양방향(PIM·GPU 동시 채움) 확인.
  - **T-S (대조군)** — short, idle 불변 확인 (과잉 수정 방지).
- [ ] **before/after idle (독립 run 운영)** — 트레이스마다 합류 OFF/ON
      (`measure_steady --no-join` vs 기본). **각 조합 = 독립 run · 독립 라벨 파일**
      (`steady_<label>.txt`), **순차 실행, 결과 나올 때마다 개별 보고**. 먼저
      `TDEC_off`/`TDEC_on` 가늠 → run 단가 확정 + 효과 확인 후 T-M·스윕으로 확장 (헛돌이 방지).
      합격: T-DEC GPU idle↓(주), T-M 양방향↓, T-S 불변.
- [ ] before/after 보고서 작성 (`REPORT_step5.md`)
- [ ] **배치_생애.md 갱신** — 스윕 확정값 반영: 배치 크기 상한, 합류 게이트 θ_high(+
      hysteresis θ_low) 최종값을 문서의 "세 한계" / §5 게이트에 구체값으로 명시.
      (현재는 기호·예시값 → 스윕 후 확정값으로)

> **STEP 5.5 선결 — idle 측정 항목(T-L/T-M/T-S·스윕)은 STEP 5.5 후 재개.** STEP 5
> 측정 시도 중 race 진단(`diag_join_race.py` → `race_result.txt`)으로 단일 mb 독점의
> 잔여 원인이 드러남(아래).

### STEP 5.5 — per-mb KV 예산 (F2 staggering 복원) ※ STEP 5 측정 중 발견
> 발견: STEP 1(seq 상한)은 **short-context 다중화만** 풀었고, **long-context 는 한 mb 가
> KV캐파(4M) 전체를 독점**(42req=3.99M)해 mb 1 형성 불가 → window=1. 거기에 합류가 풀린
> KV 를 그 mb 에 backfill 해 단일 mb 영구화. `diag_join_race.py` 확증: 동일 트레이스·KV캐파
> 에서 합류 OFF=mb 2개 / ON=mb 1개. **ARCH 정합 위반** — §5.6/F2 double-buffering 은
> "mb M 의 PIM attention 중 GPU 가 mb **M+1** QKV 처리" = 서로 다른 μ-batch 간 동시
> 실행이라 **≥2 mb 필수**(line 287·290·312), window={M-1,M,M+1} 3-state(line 369·436).
> 배치_생애 §세한계 "KV 를 여러 배치가 나눠 쓴다"가 *의도만 있고 강제 장치 부재*였음.

- [x] **per-mb KV 예산 = KV캐파 / 동시활성목표(2)** (4M/2 = 2M). divisor 는 window(3) 아니라
      **2** 로 확정: F2 는 동시 2개(M·M+1)면 충족, window 3번째는 빠지는 M-1 전이 여유. /3 은
      mb 가 n_sat=16 아래(long-ctx ~13req)로 작아져 sub-MFU → **/2 가 배치 포화 유지 + "2
      active + 1 여유" 정합**. `_STAGGERING_TARGET_MB=2`, window.capacity 로 clamp(F2 ablation
      cap=1 시 분모 1 → 단일 mb). 의도("나눠 쓴다")의 강제화.
- [x] **admission.layer1 적용** — `max_mb_kv_tokens` 한도까지만 admit (빈 mb 첫 req 예외 = 단일
      거대요청 starvation 방지). 초과분 defer → 다음 tick 에 mb 1·2 형성.
- [x] **`_try_join` 적용** — 합류도 그 mb 의 KV 예산 한도까지만 (다른 mb 몫 잠식 차단, race 해소).
- [x] 단위 테스트 — layer1 분할·첫 req 예외 + _try_join per-mb cap. admission+prefill_join **43 passed**.
- [x] tiny trace window 확증 — `diag_join_race`: 합류 ON 이 mb=1·window=1 → **mb=2·window=2** 회복.
- [ ] 정합성 회귀(KV no-leak·invariant·lifecycle e2e) 통과 → **per-mb 예산 독립 커밋**.
- [ ] 문서 갱신 — 배치_생애 §세한계에 "per-mb KV 예산(KV캐파/2)" 네 번째 강제 명시, REPORT 에 race 발견 기록.

> 미수정 영역(RTL·evaluator·trace 생성 등)은 회귀 스킵.

## 7. 산출물

- `debug_phase1/data/` — 합성 CSV (T-S/T-L/T-M + 압박 + 직렬검증 tiny)
- `debug_phase1/*.py` — 생성·계측 스크립트 (generate_traces, gen_long_pressure,
  analyze, trace_single_mb, prove_no_join, prove_serial_fast, config_small_cap)
- `debug_phase1/REPORT_baseline.md` — baseline 관측·근본 원인·수정 설계
- `debug_phase1/serial_result.txt` — 직렬 처리 검증 결과
