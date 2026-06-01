# Phase-2 측정 기록 — idle floor 증명 (2026-06-02)

> 단일 기준 = [OPERATING_POINT.md](../../OPERATING_POINT.md). 측정 substrate = [measure_steady.py](measure_steady.py),
> floor 증명 = [../analysis/floor_proof.py](../analysis/floor_proof.py).
> **핵심 결론: 측정 idle 은 "이 워크로드 + 이 알고리즘"의 floor 이며, 그 floor 의 모든 구성요소가
> 정량적으로 설명된다 (미설명 잔여손실 0).**

---

## 1. 측정 동작점 (sweep_D_longdec — 대량-상주 decode 풀)

`measure_steady.py --trace data/sweep_D_longdec.csv --seed-pool 2000` (수렴 정지):

| 항목 | 측정 | 타깃 (OPERATING_POINT §1) |
|---|---|---|
| decode 개수 | 122 | 123 |
| Σ decode KV | 12.33M | 12.3M |
| prefill 토큰 | 256 | 256 |
| prefill depth-work | 27.10M | 25.6M |
| idle GPU-A | **8.01%** | — |
| idle PIM | 12.64% | — |
| idle FFN | 12.61% | — |
| **spread** (max−min) | **4.62%** | ~0 (명중 시) |

직렬 실행(overlap 없음) 시 자원 idle ~67% 대비 1/14 — F2(double-buffering)·F3(inter-instance
pipeline) 발현. 배치 구성은 타깃 4개에 명중(decode 개수·Σkv·prefill 토큰; depth-work 만 +5.8%).

## 2. 이론 floor 산출 — 측정 batch 에 op-time 함수 직접 호출

세 자원은 각각 **단일 서버**(dispatcher `gpu_busy`·`pim_busy`·`instance_b_busy` = I4/I5/I6
불변식, 동시 1 op). steady-state 처리율은 가장 바쁜 서버의 per-μ-batch work 가 律속한다.
`floor_proof.py` 는 측정이 **실제 디스패치한 live μ-batch** 에 디스패처와 *동일한 op-time 함수*
(`compute_gpu_op_time_s`·`compute_ffn_op_time_s`·`PIMExecutor.op_time`, TP=8)를 직접 호출 →
합성 재구성 오차 0.

```
per μ-batch (한 layer, µs):
  t_gpuA = QKV 6.01 + PREFILL_ATTN 42.08 + O_PROJ 4.80 = 52.89   ← 병목
  t_pim  = DECODE_ATTN(Σkv 12.33M)                      = 50.43
  t_ffn  = FFN(batch 378)                               = 50.45
완벽 overlap → cycle = max = 52.89 µs,  floor idle_r = 1 − t_r/cycle
```

| 자원 | 이론 floor | 측정 idle | overlap gap |
|---|---|---|---|
| GPU-A (QKV+PREFILL_ATTN+O_PROJ) | **0.00%** | 8.01% | 8.01% |
| PIM (DECODE_ATTN) | 4.65% | 12.64% | 7.99% |
| FFN (Instance B) | 4.61% | 12.61% | 7.99% |
| **spread** | **4.65%** | 4.62% | — |

**핵심 일치 두 가지:**
1. **이론 floor spread 4.65% ≈ 측정 spread 4.62%.** 측정 spread 는 알고리즘 여유가 아니라 이
   batch 의 본질적 3자원 불균형(t_gpuA 52.9 vs t_pim/t_ffn 50.4). 알고리즘은 floor 에 도달.
2. **overlap gap 8.0% 가 세 자원에 균일.** 병목(GPU-A)은 이론 floor 0 이므로 측정 8% 가 곧
   overlap gap = pipeline fill/drain + 2-active staggering 전이 틈. 균일 → spread 를 안 키움.

즉 **측정 idle = 이론 floor + 균일 overlap gap(8%)** 으로 완전 분해. greedy dispatch 가 floor 에
근접함이 정량 입증된다.

## 3. floor spread 의 출처 = prefill depth-work 오버슈트 (item 3)

t_gpuA 의 **80% 가 PREFILL_ATTN**(42.08 µs). PREFILL_ATTN ∝ depth-work 이므로 GPU-A 가 병목인
직접 원인은 depth-work 가 타깃 25.6M 을 +5.8% 초과(27.10M)한 것.

**counterfactual:** depth-work 가 정확히 25.6M 이면 t_gpuA 50.56 ≈ t_pim 50.43 ≈ t_ffn 50.45
→ **floor spread 4.65% → 0.26%** (거의 완전 균형). 즉 floor spread 의 대부분이 이 오버슈트 기인.

## 4. 오버슈트의 원인 = age-cap 강제, 풀 고갈 아님 (item 4)

prefill 토큰은 256 고정(2의 거듭제곱·커널 친화, FFN batch 379 의 ③ knob). depth-work 만 가변.
오버슈트가 "풀에 깊은 것만 남아 불가피"(a)인지 "age-cap 강제 산물"(b)인지 — `--age-cap` ablation:

| age_cap | depth-work | 이론 floor spread | 측정 spread | 비고 |
|---|---|---|---|---|
| **2** (기본, 공정성 ON) | 27.10M (+5.8%) | 4.65% | 4.62% | 대기 ≤ age_cap+1, starvation 0 |
| **∞** (순수 steering) | 25.73M (+0.5%) | 0.40% | **0.11%** | depth 깊은 것 starvation(대기 37) |

**판정 = (b).** age_cap=∞ 면 순수 steering 이 depth-work 를 25.73M(타깃 명중)에 맞춘다 →
**얕은 재료는 풀에 있었다**(item-3 진단: 이상깊이 100K 보다 얕은 후보 평균 1.3, 풀 최소깊이 57K).
age_cap=2 의 오버슈트는 오래 기다린(prefill_wait≥2) **깊은 요청 강제 포함** + 토큰 개수 256 고정
탓에 깊은 토큰이 steering 의 얕은 선택을 밀어낸 결과.

**인과 사슬:**
```
age_cap=2 → 깊은 long-wait prefill 강제(개수 256 고정) → depth-work +5.8%
          → PREFILL_ATTN ↑ (t_gpuA 의 80%) → GPU-A 병목 → floor spread 4.65%
age_cap=∞ → 강제 없음 → steering 25.73M 명중 → spread 0.11% (단 starvation)
```

→ **floor spread 4.6% 는 풀 고갈이 아니라 age_cap=2 의 *공정성 비용*** (의도된 latency·공정성 ↔
idle trade-off, OPERATING_POINT §3 sweep). 끄면 idle 0.11% 까지 내려가나 starvation 발생.

## 5. 종합 — floor 도달 확정

측정 idle 의 모든 구성요소가 설명된다:

```
측정 spread 4.62% = 이론 floor spread 4.65%
                  = (공정성 비용: age_cap=2 의 prefill 오버슈트) 4.4%
                  + (워크로드 본질 불균형, age_cap=∞ 잔여)         0.26%
측정 idle 절대값  = 이론 floor + 균일 overlap gap 8.0%(fill/drain·staggering 전이)
미설명 잔여손실   = 0
```

⇒ **이 워크로드에서 측정 idle 은 알고리즘 floor 이며, floor 자체는 (공정성 비용 + overlap gap)
으로 완전 분해**된다. 알고리즘이 더 짜낼 수 있는 여지는 공정성을 포기(age_cap↑)할 때만 존재 —
즉 "더 낮출 수 없음"이 아니라 "더 낮추려면 starvation 을 받아들여야 함".

---

## 부록 A. prefill sweep 정정 (옛 REPORT 의 "512만 균형" 오류)

옛 기록은 *decode-KV 를 ~25M 에 고정한 채 prefill 만 올려* "512 외 불균형(1024+ spread 44~72%)"
이라 결론했으나, 이는 측정오류다. **각 prefill 은 *자기 균형점*(decode-KV·N_dec 동반 스케일)에서
모두 ctx 100K 균형**(OPERATING_POINT §5.2):

| prefill | 균형 ctx | X(µs) | N_dec | decode-KV | prefill-KV-work | spread% |
|---|---|---|---|---|---|---|
| **256** | 100K | 51 | 123 | 12.3M | 25.6M | 0.76 |
| 512 | 100K | 101 | 247 | 24.7M | 51.2M | 0.62 |
| 1024 | 100K | 203 | 494 | 49.4M | 102.4M | 0.63 |
| 2048 | 100K | 406 | 991 | 99.1M | 204.8M | 0.39 |

균형 ctx 는 prefill 무관 100K 고정(하드웨어 상수, §4 삼중균형서 prefill 약분). prefill 은 X·배치
규모만 스케일. **256 채택**(TBT·HBM 반감, TTFT·throughput 동손 0); 512 는 FFN MFU 포화 불가
판명 시 대안. 옛 "512만 가능"은 폐기.
