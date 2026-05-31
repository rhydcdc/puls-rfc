# Phase-1 Baseline 관측 보고 (수정 전)

## 1. 실행 조건

- Trace — `data/trace_long_min.csv` (축소 T-L). 4 요청, prefill 61K–134K (전부
  ctx > 56K), decode 24–48, 도착 ts 0.3–1.2.
- Config — `default_dummy_config` (Llama-3 70B, L=80, KVcap 4M, window cap 3).
- Output — `baseline_long_min/report.{json,md}`.

## 2. 결과

| 지표 | 값 |
|---|---|
| gpu_instance_a idle | 0.03% |
| **pim_instance_a idle** | **99.61%** |
| gpu_instance_b idle | 0.00% |
| 총 dispatch event | 244,160 |
| 노드별 dispatch | QKV / PREFILL_ATTN / DECODE_ATTN / O_PROJ 각 61,040 |
| **distinct micro-batch id** | **1 (id=0)** |
| PIM 첫 dispatch | ts ≈ 75 (시작 직후) |
| PREFILL_ATTN 마지막 dispatch | ts ≈ 2.48e7 (전체의 99.99% 지점) |

## 3. 관측 사실

1. **PIM은 매 cycle 호출되었다** — DECODE_ATTN 61,040회 = 전 cycle. idle 99.61%는
   "호출 부재"가 아니라 "호출당 op_time 미미"의 결과. 매 cycle decode-attn이 작은
   KV만 처리.

2. **prefill이 종료 직전(99.99%)까지 잔존** — PREFILL_ATTN이 끝까지 dispatch됨.
   prefill 61K–134K가 chunk 단위로 잘려 전 구간 GPU PREFILL_ATTN(O(n²))이 cycle을
   지배. 따라서 GPU bound·PIM idle은 *이 트레이스에서는 정상 귀결*.

3. **마이크로배치가 평생 1개** — 4 요청이 첫 admission tick에 모두 한 mb로 묶였고,
   이후 새 mb 생성도 신규 합류도 없이 그 mb만 recompose 반복. window capacity 3을
   전혀 활용하지 못함.

## 4. 해석

- "ctx > 56K → PIM bound" 분석은 *decode 단계 한정*이다. prefill 진행 중에는
  PREFILL_ATTN이 지배하여 GPU bound가 된다. 축소 T-L은 decode가 짧고 prefill이
  거대하여 생애 대부분이 prefill 단계에 머문 결과 PIM idle이 관측되었다.
- 더 근본적으로, **balance 4-factor가 발현될 구조적 전제 자체가 부재**하다. mb가
  1개로 고정되어 cross-mb staggering도, 신규 prefill/decode 합류도 일어나지 않는다.
  PIM이 노는 GPU-bound 구간에서 다른 decode를 끼우거나, 반대 상황에서 prefill을
  끼우는 경로가 작동하지 않음을 직접 확인.

## 5. 단일 mb 귀결의 코드 경로 (정밀 특정)

`trace_single_mb.py` 로 SchedulerCore 를 직접 구동·계측한 결과:

```
요청 4개 kv_length: 72032 / 98040 / 61024 / 134048  →  Σ = 365,144
KV capacity = 4,000,000  →  365,144 ≪ 4M (9% 점유)

mb 생성 이력:  ts=10.30  mb_id=0  prefill_reqs=4  decode_reqs=0   (단 1회)
ADMISSION_TICK:
  tick 1 @0.0   queue=0 in_flight=0 window=0 kv=0        (priming, 도착 전)
  tick 2 @10.3  queue=0 in_flight=4 window=1 kv=365144   (4요청 일괄 admit)
  tick 3+ @...  queue=0 in_flight=4 window=1 kv=365144   (영구 불변)
```

인과 사슬:

1. 요청 4개가 ts 0.3–1.2 에 도착, 각 REQUEST_ARRIVAL 이 admission tick reschedule.
2. ts=10.3 첫 실질 tick 에서 `admission.layer1` 이 큐 전체를 walk → KVcap 4M 에
   365K 만 점유하므로 `can_admit` 4개 모두 True → **한 spec 에 4요청 전부** → 한
   mb(id=0) 생성.
3. 직후 request_queue=0 → 이후 모든 tick 에서 `layer1` 이 `return None` → **새 mb
   생성 영구 0.**
4. mb 1개만 `_recompose_mb` 로 80층 × 토큰 무한 반복. window=1 고정, cap 3 미사용.

## 6. 진단 — 버그의 본질 분리

- **1차 (구조적):** admission 이 KVcap 이 허락하는 한 가용 요청을 *전부 한 mb 로*
  묶는다. KVcap(4M)이 요청 합(365K) 대비 과대하여 mb 가 1개로 귀결. 결과적으로
  cross-mb staggering·μ-batch 다중화가 원천 봉쇄됨.
- **2차 (합류 부재):** 이 트레이스는 요청이 동시 도착하여 큐가 즉시 비므로 "in-flight
  합류" 로직이 있었어도 발동 대상이 없었다. 합류 부재는 *시차 도착* 시 드러나는
  별개 문제.

즉 balance 4-factor 발현의 선결 조건은 **(a) 한 tick 의 admit 결과를 복수 mb 로
분할**하거나 **(b) mb 당 배치 크기에 상한**을 두어 동시 다중 mb 를 형성하는 것이다.
신규 합류 경로(2차)는 그 위에서 의미를 가진다.

## 7. 캐파 압박 트레이스 검증 (trace_long_pressure.csv)

축소 T-L 의 "요청 합 < 캐파" 한계를 제거하고자 캐파(4M)를 압박하는 트레이스로
재검증. N=80, prefill 59K–148K, decode 24–64, 촘촘한 도착(마지막 ts 13.67),
누적 KV 7.88M (> 4M).

`prove_no_join.py` (각 mb 의 birth request 집합 vs 이후 집합 추적) 결과:

```
총 mb 생성 수: 1                         (캐파 99.9% 압박에도 여전히 1개)
동시 최대 mb (window): 1 / capacity 3     (window 미활용)
KV used peak: 3,996,970 / 4,000,000 (99.9%)
mb 0 birth = 요청 [0..41] 42개
KV remaining: 3,030
대기 요청 최소 kv_length: 59,358 > remaining 3,030  -> admit 불가
request_queue 적체: 38,  전체 queue 적체: 81
>>> 합류 발생?: False
```

### 확정된 메커니즘 — 캐파 독점 정체

1. 첫 admission tick 이 도착해 있던 42 요청을 *전부 한 mb* 에 admit → KV 99.9% 점유.
2. 잔여 캐파 3,030 토큰. 대기 요청 최소 kv_length 59,358 → `can_admit` 전부 False.
3. 첫 mb 는 42 요청이 *모두* decode 완료해야 evict (main_loop) → 그 전엔 KV release
   없음 (completion) → 캐파 영구 점유.
4. 캐파 미해소 → 신규 admit 0 → 새 mb 0 → 큐 81개 영구 적체. 단일 mb 무한 반복.

### 진단 정정

이전 가설 "캐파 초과분이 새 mb 로 간다"는 **오류**. 실제:

> admission 이 한 tick 에 가용 요청을 캐파 한계까지 한 mb 로 몰아넣어, 그 mb 가 캐파를
> 독점하고, 완료까지 새 mb 생성이 원천 봉쇄된다.

balance 4-factor 발현의 무대(동시 다중 mb) 자체가 형성되지 않음. cross-mb
staggering 불가(window 무용) + 합류 불가(큐 적체분이 기존 mb 에도, 새 mb 에도 못 들어감)
가 단일 근본 원인 — **"한 tick = 캐파까지 한 mb"** 정책 — 에서 동시 발생.

## 7b. 직렬 처리 확증 (prove_serial_fast.py)

"새 mb 가 영원히 안 생기는가, 아니면 직렬 대기인가" 를 판별. 가속 위해 KV 캐파만
200K 로 축소(`config_small_cap.py`), tiny 트레이스(5 req × 60K, decode=2)로 완주.

```
완전 drain: True, 총 step: 7,416,591
총 mb 생성 수: 2
동시 최대 window: 1 / capacity 3        (한 번도 2개 공존 안 함)

mb register 이력:
  ts=10.1         mb=0  reqs=[0,1,2]  window=0   (캐파 200K 에 3 req=180K admit)
  ts=8,671,910.1  mb=1  reqs=[3,4]    window=0   (mb0 완전 종료 후에야 생성)

판정: 동시 다중 mb = False (max window=1), 직렬 = True
```

결론: **"영원히 못 들어옴"이 아니라 "직렬 대기"**. mb0 가 ts=10→8.67e6 동안 혼자
완주한 뒤에야 KV release → mb1 생성. 두 mb 가 단 한 순간도 공존하지 않아 window
capacity 3 이 완전 무용. mb0 가 도는 동안 PIM 이 놀아도 mb1(아직 미생성)의 prefill
로 빈자리를 못 채움 — cross-mb staggering·balance·합류가 전부 죽는 단일 근본 원인.

## 8. 수정 설계 — 세 한계의 분리

현재 구현은 KV 캐파(메모리) 하나를 배치 크기 한계로도 겸용하여 단일 mb 독점을
초래한다. 목적이 다른 세 한계를 분리한다.

| 한계 | 성격 | 역할 |
|---|---|---|
| **KV 캐파** | 하드 | OOM 방지 (인스턴스 전체 동시 보유 KV 한계, 여러 mb 공유) |
| **배치 크기 (seq 상한)** | 하드 | head-of-line blocking 방지 (한 mb 의 최대 요청 수) |
| **token budget** | 소프트 기본값 | PULS 동적성이 사는 곳 (balance 가 PIM/GPU idle 보고 동적 조절) |

- **KV 캐파 = 하드** — OOM 방지. 그대로 4M 유지.
- **배치 크기(seq 상한) = 하드** — head-of-line blocking 방지. 신설.
- **token budget = 소프트 기본값** — balance 4-factor 가 동적 조절. PULS 동적성의 핵심.
  실제 모델을 돌리지 않아도 트레이스의 decode 토큰 수 + prefill 길이로 closed-form
  산출 가능 (per-token FLOPs × batch / GPU peak, PIM 임계 56K/65536 동반).

상한과 그 안의 balance 는 직교 — 상한은 천장, balance 는 천장 아래 공간의 동적
조절. 충돌하지 않으며, 상한이 오히려 balance 가 작동할 무대를 만든다.

표준 정합: vLLM `max_num_seqs`(seq 상한) / `max_num_batched_tokens`(token budget),
Sarathi-Serve token budget(decode + chunked prefill 혼합). PULS 는 Sarathi 의 PIM
확장. 현 구현이 두 한계를 겸용한 것이 오히려 비표준.

## 9. 다음 단계

- 1차 = **배치 크기(seq 상한)** 신설 — KV 캐파와 분리. 동시 다중 mb 형성 (balance·
  staggering·합류의 공통 선결 조건).
- 배치 크기 스윕 = **{256, 512} 2값만**. 각각 idle_fraction + (대리)TTFT/TBT 관측.
- token budget = closed-form 산출(트레이스 decode/prefill 기반), balance 동적 조절 입력.
- 2차 = 신규 요청의 in-flight 합류 경로.
- 수정 후 trace_long_pressure 재실행 → mb 다중화·idle 변화 측정.
- 최종 검증 = 풀 3종(T-S/T-L/T-M): T-S 대조군 idle 불변, T-L/T-M idle 감소.
