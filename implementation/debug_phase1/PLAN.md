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

### 합성
- [ ] CSV 생성 스크립트 작성 (`debug_phase1/`)
- [ ] T-S 생성 — short 몰림, decode 고분산, 도착 포화
- [ ] T-L 생성 — long 몰림, 56K–150K, KV캐파 내 동시성 확보
- [ ] T-M 생성 — short+long 혼합, prefill 분산 실 트레이스급
- [ ] 분포 sanity (ctx/decode min·mean·max, 동시 admit 추정)

### Baseline (수정 전)
- [ ] T-S/T-L/T-M 실행, idle 3-key + dispatch_trace 저장
- [ ] T-L에서 PIM 바운드·GPU 유휴 구간 확인
- [ ] 신규 prefill 미합류 / `+1` 무실효 확증

### 수정
- [ ] 합류 경로 신설 (chunked prefill)
- [ ] 합류 경로 신설 (chunked decode)
- [ ] window capacity 한도 검토
- [ ] 기존 테스트 회귀 통과

### 재검증 (수정 후)
- [ ] T-L — GPU/종합 idle 감소 확인
- [ ] T-M — 양방향 idle 동시 감소 확인
- [ ] T-S — idle 불변 확인 (대조군)
- [ ] before/after 보고서 작성

## 7. 산출물

- `debug_phase1/data/` — 합성 CSV 3종
- `debug_phase1/` — 생성 스크립트, baseline·재검증 결과, 최종 보고서
