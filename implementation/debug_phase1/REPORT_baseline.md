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

| 한계 | 성격 | 역할 | 정하는 법 |
|---|---|---|---|
| **KV 캐파** | 하드 | OOM 방지 (인스턴스 전체 동시 보유 KV, 여러 mb 공유) | 메모리 (4M 유지) |
| **배치 크기 (seq 상한)** | 하드 | head-of-line blocking 방지 (한 mb 의 최대 *요청 개수*) | 스윕 {256, 512} |
| **window (배치 개수)** | 하드 | staggering 깊이 (동시 생존 mb 수) | **작게 고정 (3)**, 메모리로 안 잡음 |
| **token budget** | 소프트 기본값 | PULS 동적성 (balance 가 PIM/GPU idle 보고 동적 조절) | closed-form 산출 |

- **KV 캐파 = 하드** — OOM 방지. 그대로 4M 유지.
- **배치 크기(seq 상한) = 하드** — head-of-line 방지. 신설. *요청 개수* (KV 길이 아님,
  vLLM `max_num_seqs` 와 동일).
- **window(배치 개수) = 하드, 작게 고정(3)** — staggering 은 mb 2~3 개면 충분(PIM 도는
  동안 GPU 빈자리 채움 = double-buffering F2). 메모리 한계까지 늘리면(`KV캐파/배치크기`)
  동시 KV 가 다시 캐파 전체가 되어 단일 mb 독점 병으로 회귀 → **금지**.
- **token budget = 소프트 기본값** — balance 4-factor 가 동적 조절. PULS 동적성의 핵심.
  실제 모델 미실행, 트레이스의 decode 토큰 수 + prefill 길이로 closed-form 산출
  (per-token FLOPs × batch / GPU peak, PIM 임계 56K/65536 동반).

상한과 그 안의 balance 는 직교 — 상한은 천장, balance 는 천장 아래 공간의 동적
조절. 충돌하지 않으며, 상한이 오히려 balance 가 작동할 무대를 만든다.

표준 정합: vLLM `max_num_seqs`(seq 상한) / `max_num_batched_tokens`(token budget),
Sarathi-Serve token budget(decode + chunked prefill 혼합). PULS 는 Sarathi 의 PIM
확장. 현 구현이 두 한계를 겸용한 것이 오히려 비표준.

## 8b. 운영 의미론 — 배치 구성·합류·종료·밸런스

본 절은 수정의 동작 의미를 확정한다 (구현 전 합의).

### 실제 배치 크기 = 두 하드 한계의 min

```
실제 배치 크기 = min(seq 상한, KV 캐파가 허용하는 요청 수)
```

- long-context (avg kv 100K): KV 캐파 4M / 100K ≈ 40 < seq 상한 256 → **KV 캐파가 binding**.
- short-context (avg kv 10K): 4M / 10K = 400 > 256 → **seq 상한이 binding**.

→ "seq 상한 256 이어도 KV 가 먼저 차면 배치가 작아진다" 는 정상. 둘 중 빡빡한 쪽이 이긴다.

### 합류 가능량 = 두 여유의 min

```
합류 가능량 = min(seq 상한 − 현재 요청 수, KV 캐파 여유 / 신규 요청 kv)
```

- 합류 대상 = **request_queue 의 신규 요청** (다른 mb 에서 빼오는 것이 *아님*).
- 트리거 = 요청 완료 → KV release → 여유 발생 → 큐에서 끌어와 끼움.
- long-context 는 보통 KV 여유가 binding, short 는 seq 자리가 binding.

### 연속 배칭 vs 밸런스 — 직교

- **연속 배칭 (합류)** = "*누구를* 배치에 넣을까" — 자리(seq+KV) 나는 대로 큐에서 들임.
- **밸런스 4-factor** = "들어온 요청들의 *일을 어떻게 쪼갤까*" — prefill chunk 크기를
  PIM/GPU idle 보고 동적 조절.
- 합류가 넣고, 밸런스가 그 안에서 쪼갠다. 한쪽이 다른 쪽을 대체하지 않아 충돌 없음.

### 밸런스 입력 = 미래 decode 길이 아님, *현재 KV 길이*

- decode 는 매 cycle 토큰 1 개씩 (autoregressive, 쪼갤 수 없는 최소 단위).
- 한 토큰의 attention 시간 = **지금 그 요청의 KV 길이**(prefill + 기생성분)로 확정 →
  미래에 몇 토큰 더 나올지 *예측 불필요*.
- 밸런스는 매 cycle "현재 살아있는 decode 요청들 × 각자 현재 KV → PIM 시간 확정 →
  GPU prefill chunk 조절" 하는 **상태 기반 피드백 루프**. 요청이 EOS 로 빠지면 다음
  cycle 에 다시 계산 (자기 보정).
- 트레이스의 `max_tokens`(예 350)는 *종료 시점* 결정용일 뿐 밸런스 입력 아님. 실제
  서버의 가변 decode 길이/EOS 여도 밸런스 로직은 그대로 작동.
- 용어 주의: "chunked decode" 는 한 요청의 decode 를 쪼개는 것이 *아니라* 여러 요청
  decode 토큰을 한 배치에 **모으는(batching)** 것. 기존 스케줄러에 "chunked decode"
  가 없는 이유도 예측 문제가 아니라 decode 가 쪼갤 수 없는 최소 단위이기 때문.

### 종료 시점 — 요청 vs mb 구분

- **개별 요청** = `decoded_count >= max_tokens` 도달 시 확정 종료 + KV 즉시 release.
  영원히 안 끝나는 일 없음.
- **mb** = (a) 합류할 신규 요청 없음(큐 빔) **AND** (b) 안의 요청이 모두 완료 — 두 조건
  동시 성립 시 evict. 큐에 일감이 있는 한 합류로 유지 = 연속 배칭 정상 동작 (무한 아님).
- **mb 가 영원히 안 끝나는 경우** = 도착률이 처리율을 *영구* 초과해야 성립 = 시스템
  과부하(overload). 정상 운영(도착률 < 처리율)에선 큐가 주기적으로 비어 evict 됨.
  영구 초과는 mb 종료 문제가 아니라 큐 무한 적체(DDoS 급)로, admission control
  (`request_queue_capacity`)이 거부할 영역. 정상 밸런스의 고려 대상 아님.

### decode 합류의 비용 — GPU projection + KV 캐시 동반 증가

decode 요청을 PIM 빈자리에 채우는 것은 *PIM 만* 일을 받는 것이 아니다.

- decode 요청 1 개도 매 cycle **QKV + O_PROJ 를 GPU 에서** 수행 (GPU_NODE_TYPES).
  PIM 이 받는 건 decode-attn 하나뿐. → decode N 개 추가 = **GPU projection 도 N 에
  비례 증가** (요청당 ~0.2288 µs).
- decode 요청이 살아있으려면 그 **KV 캐시가 메모리에 상주**해야 함 (kv_length). decode
  를 많이 채우려면 그만큼 **KV 캐파 여유가 커야** 함 — 부족하면 캐파에 막혀 못 채움.
- 함의: "PIM 에 decode 때려박으면 idle 해소" 가 단순 성립하지 않음. GPU 가 같이 커지므로
  **컨텍스트가 충분히 길 때(>56K)** 만 PIM 증가분이 GPU 증가분을 추월 (PIM/GPU =
  ctx/56,160, N 약분). 그 긴 컨텍스트 decode 를 다수 담으려면 KV 캐파가 받쳐줘야 함.

## 9. 다음 단계

- 1차 = **배치 크기(seq 상한)** 신설 — KV 캐파와 분리. 동시 다중 mb 형성 (balance·
  staggering·합류의 공통 선결 조건).
- 배치 크기 스윕 = **{256, 512} 2값만**. 각각 idle_fraction + (대리)TTFT/TBT 관측.
- token budget = closed-form 산출(트레이스 decode/prefill 기반), balance 동적 조절 입력.
- 2차 = 신규 요청의 in-flight 합류 경로.
- 수정 후 trace_long_pressure 재실행 → mb 다중화·idle 변화 측정.
- 최종 검증 = 풀 3종(T-S/T-L/T-M): T-S 대조군 idle 불변, T-L/T-M idle 감소.

## 10. STEP 1 진행 결과 + ADMISSION_TICK 과다 self-reschedule 발견

### 10.1 STEP 1 (seq 상한 분리) — mb 다중화 확증

`max_batch_size` 신설 (config, 기본 256) + admission.layer1 에 min(seq 상한, KV
허용분) 적용 + window 3 고정. 타깃 회귀 269 passed.

인과 분리 검증 (`prove_multiplex.py`): 동일 trace(serial_tiny)·동일 KV 캐파(200K),
**seq 상한만** 다르게 →
- seq 무제한: 직렬, 동시 최대 window = 1 (baseline 재현)
- seq = 2: 다중, 동시 최대 window = 2

→ mb 다중화는 오직 seq 상한 덕. STEP 1 목표(동시 다중 mb 형성) 달성.

### 10.2 발견 — ADMISSION_TICK 이 KERNEL_COMPLETION 1 건당 수~수십 회 헛돈다

STEP 1 idle 측정 시도 중, prefill_chunk 를 크게(8192) 잡은 config 에서 완주가
극단적으로 느리고 메모리가 8.9GB 까지 증가. 정밀 추적(`diag` 류) 결과:

```
[50k step]  clk=49,900   mb0 layer=23  O_PROJ=RUNNING  큐: ADMISSION_TICK 10 + KC 1, KC_ts=49,987
[200k step] clk=199,620  mb0 layer=14  (다음 토큰으로 리셋됨 — 진행 중)
[400k step] clk=399,271  mb0 layer=19
```

- **라이브락 아님** — clock·layer 모두 꾸준히 전진, mb 는 80 층을 돌며 토큰 생성.
  (초기 "queue 11/in_flight 8 고정"만 보고 라이브락이라 본 것은 *오판*, 정정함.)
- **진짜 문제 = admission tick 과다 self-reschedule**: tick 간격(10µs) ≪ O_PROJ
  op_time(~87µs, prefill chunk 8192 × long ctx). GPU 가 한 op 도는 동안 admit 도
  불가한 상태에서 tick 이 ~9 회 헛돌며 매번 새 tick self-push
  (`_schedule_next_admission_tick`). KERNEL_COMPLETION 1 건 처리에 tick 수~수십 건
  낭비 → step·메모리 폭증.

### 10.3 왜 고쳐야 하는 실제 버그인가

- prefill chunk 8192 는 측정 편의로 넣었으나 **실제로 발생 가능한 값**: (a)
  `balance_pim_slack` 이 PIM-bound 시 chunk 를 동적으로 키움(ARCH 의도), (b)
  production token budget 도 보통 2048–8192 (vLLM/Sarathi). balance·합류가 강해질수록
  chunk 가 커져 더 자주 터짐.
- 근본 원인 = **admission 을 고정 시간 간격(tick_interval_us) 타이머로 self-reschedule**
  하는 설계. GPU 가 긴 op 로 바쁜 동안 진전 없는 tick 이 누적. 표준 스케줄러
  (vLLM/Sarathi)는 admission 을 별도 타이머가 아니라 *iteration(배치 step) 경계*에서
  호출하므로 이 문제가 없음.
- STEP 3(합류)도 admission 경로에 얹히므로 **STEP 3 전에 선결**해야 함 (→ STEP 2.5 신설).

### 10.4 STEP 2.5 수정 — 이벤트 기반 admission (완료)

(b)안 채택 — admission 을 고정 타이머가 아니라 *이벤트 경계*에서만 재기동:

1. **KERNEL_COMPLETION 에 admission 시도 추가** (main_loop) — 완료 = 자원이 비는
   유일 시점 = admit 기회 (vLLM/Sarathi 식 iteration-boundary admission).
2. **고정 타이머 self-push 제거** — `_schedule_next_admission_tick` 함수 자체 삭제
   (orphan). ADMISSION_TICK 처리 후 다음 tick 을 +interval 에 self-push 하던 것 폐기.
3. **REQUEST_ARRIVAL 트리거 유지** — idle 후 새 요청 도착 시 cold-start 재기동.

재기동 단일 경로 = `_schedule_admission_tick_with_default_payload`
(KERNEL_COMPLETION / REQUEST_ARRIVAL 에서만 호출).

### 10.5 효과 — 헛도는 tick 제거 확증

light_pressure 트레이스 (이전엔 prefill chunk 8192 로 안 끝나던 것) 재실행:

```
수정 전: HIT LIMIT 3,000,000 (not drained), 메모리 8.9GB
수정 후: DRAINED at step 76,820  (완주, ~40× 감소)
```

→ 완주 불가의 원인이 admission tick 헛돌이였음이 확정. 타깃 회귀 284 passed
(TestSelfRescheduling 4 건은 타이머 전제 → 이벤트 기반으로 업데이트).

## 11. STEP 3 합류 전 before 수치 — 풀 트레이스 (trace_long_pressure, 80 req)

STEP 2.5 후, 이전엔 완주 불가하던 80 req 압박 트레이스를 default config 로 측정:

```
drain=True  steps=5,740,320  mb_count=4  max_window=3/3
idle:  gpu_instance_a=0.04%   pim_instance_a=97.95%   gpu_instance_b=0.00%
```

관측:
- **완주** — 이전(3M step not drained, 8.9GB) 대비 STEP 2.5 효과로 정상 종료(메모리 <1GB).
- **mb 다중화 작동** — max_window 3/3 (window cap 까지 참).
- **PIM idle 97.95%** = STEP 3 합류 전 before 수치. GPU A 만석, PIM 거의 유휴.
  → 이 트레이스는 prefill-dominant(decode 24–64 로 짧음)라 PIM 채울 decode 일감이 희소.

해석:
- 이 트레이스에선 합류해도 PIM idle 이 크게 안 떨어질 수 있음 (decode 일감 자체가 적음).
- **합류 효과는 decode 비중 큰 트레이스(T-M / T-L)에서 드러남**. STEP 3 후 after 측정은
  T-M/T-L 로 (decode 분산 높고 long-ctx → PIM-bound 구간 형성).
- mb_count=4 는 KV 캐파(4M)가 seq 상한(256)보다 binding 이라 정상 (long-ctx).

## 12. STEP 5 재검증 착수 — race 발견 + per-mb KV 예산 (STEP 5.5)

STEP 5(수정 후 idle 측정) 착수 중 측정 방법과 잔여 버그 두 가지가 드러났다.

### 12.1 측정 프로토콜 — 완주 대신 수렴 조기종료 (엄밀판)

완주는 (a) 비용 과대(decode×80층), (b) 막판 도착 끊긴 ramp-down 꼬리가 idle 오염.
대신 *워밍업(전 도착 주입까지) → idle_telemetry.reset → 정상상태에서 누적 idle 수렴
(Δ<0.005)하면 정지*. before/after·스윕에 동일 프로토콜(`measure_steady.py`).

### 12.2 race 발견 — 합류가 freed KV 를 backfill 해 단일 mb 독점

축소 T-L·long_pressure 측정에서 합류 ON 인데도 mb_count=1·max_window=1 관측. 진단
(`diag_join_race.py`, 동일 트레이스·KV캐파, 합류만 on/off):

```
join=OFF  mb_count=2     (요청 완료로 KV 풀리자 admission 이 mb 1 생성)
join=ON   mb_count=1     (같은 완료 이벤트의 _try_join 이 freed KV 를 mb 0 에 먼저
                          backfill → admission tick(+10µs) 도착 시 KV·큐 비어 mb 1 불가)
```

→ 합류가 매 완료마다 freed KV 를 가로채 단일 mb 로 수렴. **ARCH §5.6/F2 정합 위반** —
F2 double-buffering 은 "mb M 의 PIM attention 중 GPU 가 mb M+1 QKV 처리" = 서로 다른
μ-batch 동시 실행이라 ≥2 mb 필수. 단일 mb 면 F2 무대 자체가 없음.

### 12.3 근본 — per-mb KV 미강제 (STEP 1 은 short-context 만 해결)

STEP 1(seq 상한)은 short-context 다중화만 풀었다. long-context 는 한 mb 가 KV캐파(4M)
전체를 독점(예 42req=3.99M)해 mb 1 형성 불가 → window=1. 배치_생애 §세한계 "KV 를
여러 배치가 나눠 쓴다"가 *의도만 있고 강제 장치 부재*였음.

### 12.4 수정 — per-mb KV 예산 = KV캐파 / 동시활성목표(2)

per-mb KV 예산 = KV캐파 / 2 (= 2M) 을 admission.layer1 + `_try_join` 양쪽에 적용.

- **분모 = 2** (window 3 아님): F2 는 동시 2개(M·M+1)면 충족, window 3번째 슬롯은 빠지는
  M-1 전이 여유. /3 은 mb 가 n_sat=16 아래(long-ctx ~13req)로 작아져 sub-MFU →
  **/2 가 배치 포화 유지 + "2 active + 1 여유" 정합**. `_STAGGERING_TARGET_MB=2`,
  window.capacity 로 clamp(F2 ablation cap=1 → 분모 1 → 단일 mb).
- **layer1** — mb 가 제 예산까지만 admit (빈 mb 첫 req 예외 = 단일 거대요청 starvation 방지),
  초과분 defer → 다음 tick 에 mb 1·2 형성.
- **_try_join** — 합류도 그 mb 예산 한도까지만 backfill (다른 mb 몫 잠식 차단, race 해소).

재확증 (`diag_join_race`, per-mb 예산 적용 후): 합류 ON 이 **mb=1·window=1 → mb=2·
window=2** 로 회복. F2 동시 활성 2개 달성. 단위 43 passed, 빠른 정합성 회귀 61 passed
(KV no-leak·invariant·완료·integration·f4).

## 13. balance_pim_slack 단위 버그 (ns vs µs) 발견·수정 — agentic 측정 중

STEP 5 현실적 agentic 트레이스(ctx>56K + 긴 decode) 측정 중, 합류 ON 에서 GPU 가
극단 과포화(gpu_a idle≈0, pim_a idle≈98%)되는 현상을 추적해 발견.

### 13.1 버그 — admission balance 경로의 ns/µs 불일치

- `PIMExecutor.op_time()` 은 **ns** 반환 (tile_time_ns 기반).
- dispatcher 는 실행 타이밍에서 PIM op_time 을 `× 1e-3`(ns→µs)로 정정해 GPU(초→µs)와
  일관 (dispatcher._op_time, Stage 2 정정).
- **그러나 admission balance 경로는 그 정정을 누락**: `_make_t_pim_fn` 이 op_time 을 ns
  그대로 반환 → `balance_pim_slack` 이 `gpu_slack_us = t_pim(ns)×margin − t_proj(µs)` 로
  **ns 와 µs 를 뺌** → t_pim 이 1000× 부풀어 `chunk_optimal` 과대(예 ~7900 토큰) →
  prefill chunk 예산 폭증 → **GPU 가 prefill 로 과포화, PIM 유휴**.

### 13.2 수정 — `_make_t_pim_fn` 에 ×1e-3 (ns→µs)

dispatcher 와 동일 convention 적용. 한 줄. 단위테스트 신설
(`test_compose_payload_t_pim_fn_returns_microseconds` — fn(n) == op_time(ns)×1e-3 고정).
타깃 회귀(admission/chunk/payload-wiring/main_loop/integration) 68+12 passed.

### 13.3 측정 단서 — agentic OFF/ON (warmup decode_frac 0.9, 순수 고ctx decode)

`trace_agentic` (prompt ~69K 전부 >56K, decode ~20K, N=30):

| | gpu_a idle | pim_a idle | 해석 |
|---|---|---|---|
| OFF (합류X) | 0.343 | 0.458 | 순수 decode → **PIM 활용(유휴 98%→46%)**, 둘 다 ~40% |
| ON (합류O, **버그 전**) | 0.002 | 0.977 | GPU 과포화·PIM 유휴 (단위 버그 영향) |
| ON (합류O, **버그 수정 후**) | 0.002 | 0.977 | 버그 전과 **bit-identical** (수정 무효) |

- **확정**: 순수 고ctx decode 에서 PIM 은 실제로 bound·활용됨(OFF pim idle 46%). 이전
  관측 PIM idle 98% 는 prefill flood 구간의 산물이었음.
- **단위 수정은 agentic 측정에 무효**(bit-identical): 고ctx 혼합은 t_proj 가 커서
  `t_pim×margin − t_proj < 0` → chunk_optimal=0 → `max(base 512, 0)=512` 로 귀결, t_pim
  이 ns/µs(1000×차)든 둘 다 음수라 동일. 즉 agentic GPU 과포화의 원인은 단위 버그가
  아니라 **(a) base chunk 512 의 고ctx prefill flood + (b) join 의 backlog prefill 충전
  (연속배칭 throughput 동작)**. 단위 수정은 양의 슬랙 regime 에서 발현하는 별개 버그라 유지.

### 13.4 재해석 — ON 은 의도대로 동작 (throughput-max)

ON 의 GPU 유휴 0.34→0.002 는 **합류(연속배칭)가 GPU 유휴를 backlog prefill 로 메운
가치**다. PIM 유휴가 높은 건 decode-attn 이 본질적으로 싸서(물리)이지 버그 아님.
"둘 다 낮은 유휴"는 backlog 가 있는 한 구조적으로 불가 — join 이 GPU throughput 을
우선하는 것이 정상. PIM 의 가치는 가동률이 아니라 op-level(Aux2 버스 절감·F5).

## 14. 스케일 스펙트럼 — 모든 규모에서 메커니즘 건전 (단위 수정 후)

전 스케일에서 스케줄러가 건전 동작하는지 검증 (T-S 최단 / T-GEN 일반 / agentic 고ctx).
전부 default config (chunk 512, θ 0.3/0.05, batch 256).

| trace | ctx | gpu_a idle | pim_a idle | mb / window | 해석 |
|---|---|---|---|---|---|
| T-S ON | ~8K | 0.0002 | 0.9943 | 3 / 3·3 | 저ctx → GPU-bound (PIM 일감 微, 정상) |
| T-GEN ON | ~13K | 0.0002 | 0.9935 | 3 / 3·3 | 저ctx → GPU-bound |
| T-GEN OFF | ~13K | 0.0010 | 0.9913 | 3 / 3·3 | OFF≈ON (저ctx 합류 무관) |
| agentic OFF | ~70K+ | 0.343 | 0.458 | 3 / 3·3 | 순수 decode → **PIM 활용(유휴 46%)** |
| agentic ON | ~70K+ | 0.002 | 0.977 | 3 / 3·3 | join 이 GPU throughput 충전 |

결론:
1. **메커니즘은 스케일 무관 건전** — 모든 trace 에서 mb=3·window 3/3 (staggering·다중 mb·
   per-mb 예산·합류 작동), 과포화·라이브락 0, GPU 잘 활용.
2. **PIM 활용은 물리(ctx/56,160)를 따름** — <56K 면 PIM 유휴(decode-attn 싸서, 정상),
   >56K 순수 decode 면 PIM 활용(46%). ARCH 56K 임계와 정확히 일치.
3. **join = GPU throughput 가치** — GPU 유휴가 존재하는 곳(agentic OFF 34%)을 backlog
   prefill 로 메움(→0.2%).
4. 저/중 스케일 PULS 가치는 PIM 가동률이 아니라 다른 가속원(인스턴스 분리·Aux1 등);
   PIM 특화 가치는 고ctx(타깃 agentic 장기추론)에 집중 — 타깃 정합.

> 산출물: `measure_steady.py`(엄밀판 측정), `gen_agentic.py`·`gen_general.py`·
> `gen_step5_traces.py`(트레이스), `diag_join_race.py`·`diag_mb_timeline.py`·
> `diag_optime.py`(진단), `steady_*.txt`·`race_result.txt`(결과).
