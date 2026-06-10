# PULS — PIM-Unified LLM Serving

**Scheduler-aware co-design of HBM-PIM and production LLM serving stack.**

> **Disclosure** — 학부생 단일 저자의 개인 연구 프로젝트. 소속 기관·vendor 연계 없음.
>
> 혼자 공부하며 배우는 과정이며, 피드백 · 지적 · 가르침을 적극 환영합니다 (GitHub Issues / Discussions).

📊 **시각 자료:** [**rhydcdc.github.io/puls-rfc/scheduler_policy.html**](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html)

본 repo 는 진행 중인 prototype 의 공개 RFC (Request for Comments). Architecture + design rationale + scheduler 정책의 entry point 와 함께, real long-context trace 위 4 가속 source 의 calibrated projection 산출도 함께 제공 ([Results](#results) 참조).

상세 본문 — [`ARCHITECTURE.md`](ARCHITECTURE.md) (substrate, instance disaggregation, scheduler integration, adaptive admission, layer flow, prior art 비교 통합).

> **업데이트 일지.**
> - **1차 업그레이드 — 스케줄러 로직 일반화.** 모델·HW 변수화 `derive` + 단일 CONTRACT 아래 `runtime` / `sim` / `validation` 분리 ([`puls-engine`](puls-engine/CONTRACT.md), 196 checks). 동작점은 hand-set 이 아니라 *도출*된다.
> - **2차 업그레이드 — 글로벌 스케줄러 + 인스턴스 종속성 모델.** **글로벌 age-cap**(노드 간 FIFO 공정성 → 강제 라우팅)과 **on-node 멀티턴 KV 캐시**를 추가하고, TBT 에 **인스턴스 내/간 종속성**을 모델링(Instance A→B 종속성 + PIM↔GPU-A HBM 경합). 승리한 클러스터 설계는 **C**, config 와 수치는 [클러스터 스케줄러 (C)](#cluster-scheduler-c).
> - **3차 업그레이드 — 물리적 캐시 어피니티 + 정직한 회계.** 멀티턴 복귀를 이제 KV 를 HBM 에 물리적으로 보유한 노드로 라우팅하고(어피니티 대기열, hole 발생 시 1순위; 글로벌 age-cap 25 와 별도의 spill cap 200 — 캐시된 복귀에게 기다림은 보상이 있다), 캐시 예산은 멀티턴 풀-KV 팽창분(실측 ~0.24 TB)만큼 동적으로 차감하며, SSD reload 상수를 현실적 NVMe-array 값으로 정정(2e7 → 1e8 B/round; 옛 값은 recompute 손익분기 2.1e7 보다 5% 낮아 tier 2 가 수학적으로 무용했음), per-request 토큰 간격 KPI([PAUSE])로 배치 레벨 TBT 의 일시정지 비용 사각을 메운다. 물리적 캐시 명중 1.7% → 99.7%. 수치는 [클러스터 스케줄러 (C)](#cluster-scheduler-c).

## 목차

**Background**
- [Processing-in-Memory 아키텍처의 특징](#processing-in-memory-아키텍처의-특징)
- [문제 의식](#문제-의식)

**The Proposal**
- [PULS 의 제안](#puls-의-제안)
- [접근 요약](#접근-요약)
- [워크로드 적용 범위](#워크로드-적용-범위)
- [가속 Source 요약](#가속-source-요약)
- [Mixed Batching 의 역할](#mixed-batching-의-역할)

**Results**
- [Results](#results)
- [Runtime Validation](#runtime-validation)
- [클러스터 스케줄러 (C)](#cluster-scheduler-c)
- [Limitations / Disclosure](#limitations--disclosure)

**Resources**
- [Interactive Reading Guide](#interactive-reading-guide)

## Processing-in-Memory 아키텍처의 특징

*본 § 는 모든 PIM 아키텍처가 공유하는 구조적 제약을 정리.* HBM4E 부터 custom logic die 공정이 시작된다. PIM 이 기존 GPU 처리와 병렬로 MAC 연산을 수행하려면 MAC 유닛을 **HBM 의 logic die (PHY) 또는 DRAM die** 에 설치해야 한다. 이 substrate 선택이 다음 구조적 제약을 야기:

- **Logic die (PHY) 설치 시 — 면적 제한** — Logic die 의 가용 면적이 제한적이라 의미 있는 양의 연산기 (general-purpose GEMV / GEMM scale) 를 설치할 수 없음.
- **DRAM die 설치 시 — 메모리 용량 감소** — DRAM die 에 MAC 을 추가하면 메모리 셀 영역이 잠식되어 가용 KV cache 용량 감소.
- **Thermal envelope 제약** — DRAM 은 열에 민감하여 대량의 MAC 연산을 sustain 못 함. PIM 활성화 구간에서 throttling 이 필연적으로 동반.
- **범용 연산 지원 비현실** — 위 3 제약의 결합으로 모든 범용 연산을 PIM 으로 흡수하는 것은 비현실. PIM scope 의 신중한 한정이 필수.
- **메모리 path 공유 — GPU ↔ PIM 동시 점유 불가** — 메모리 데이터는 GPU 와 PIM 코어가 동시 점유 불가. 양 측이 반드시 겹치는 통로 (TSV 또는 bank 간 path) 를 지나야 하며, 이는 GPU 측 데이터 load 의 지연 요인.

## 문제 의식

기존 HBM-PIM 연구들의 한계, 축별 분류:

**Substrate-level (cell · bank · die)**
- **메모리 die 침범** — PIM logic 이 메모리 die 영역을 잠식하여 KV cache 가용 용량 감소.
- **발열 / thermal envelope 제약** — PIM 활성화 구간에서 thermal throttling 으로 PIM 동작 중단.
- **비표준 DRAM 회로 수정 요구** — dual row buffer, bank-level compute logic 등 cell · bank 수준의 회로 변경은 표준 DRAM fab 공정에서 벗어나며, 일반 logic ASIC 통합을 넘어서는 fab 위험·수율 부담을 부과.

**System-level (interconnect · resources · auxiliary HW)**
- **복잡한 부가 스케줄링 로직 + 추가 하드웨어 모듈** — bespoke 제어기 · DMA 엔진 등 substrate 외 추가 영역 요구.
- **Logic die ↔ DRAM die / HBM ↔ host 왕복 트래픽** — attention 중간 결과 (softmax accumulator, row max 등) 가 logic die 와 DRAM die 사이, 그리고 HBM 과 host (GPU / NPU) 사이를 왕복하여 internal bus 와 host ↔ HBM 경로를 점유.
- **GPU 외 별도 대용량 HBM 자원의 구조적 요구 (일부 설계에서)** — PIM 측 HBM 을 GPU HBM 과 물리적으로 분리하는 설계는 호스트 GPU 메모리에 더해 별도 대용량 HBM 자원을 추가로 요구.

**GPU ↔ PIM concurrency (stall · path contention)**
- **PIM 실행 중 GPU stall (일부 설계에서)** — 일부 설계는 PIM 이 동작을 완료할 때까지 GPU 를 명시적으로 block 하여, 병렬 자원이어야 할 두 영역을 직렬화. 이 구간의 GPU idle 시간은 다른 작업으로 채울 수 없음.
- **공유 메모리 path 경합** — PIM 과 GPU 는 동일 물리 경로 (TSV / C/A bus / inter-bank path / row buffer) 를 공유. PIM 활성화는 GPU 의 동시 HBM read (weight streaming, KV 트랜잭션) 와 경합하여 GPU 측을 지연.

**Serving lifecycle (KV cache continuity · session state)**
- **멀티턴 / continuation prefill 미지원 (다수의 기존 설계에서)** — 기존 설계는 통상 request 당 single Sum → repeated Gen 흐름 가정. 기존 KV 를 다시 attend 해야 하는 continuation prefill (멀티턴 채팅, chunked prefill, context 가 누적되는 RAG 등) 은 구조적으로 수용 불가.
- **GPU ↔ 별도 PIM 자원 간 KV 대량 이동 (일부 설계에서)** — PIM 측 HBM 이 GPU 와 물리적으로 분리된 경우, KV cache 가 host ↔ PIM interconnect (PCIe / NVLink / CXL) 를 통과해야 하며, 이 interconnect 가 새 병목이 되어 PIM 의 internal-bandwidth 이점을 부분적으로 무효화.

**Workload coverage (model variants · batch regime)**
- **모던 서빙 기능과 비호환** — GQA, speculative decoding 등 production serving 의 표준 기능 지원 부재.
- **배치 크기별 성능 우위 불안정성 (다수의 기존 설계에서)** — batch size sweep 에서 PIM 우위 영역이 좁고 transition 이 비연속.
- **PIM attention 자체의 산출량 한계** — attention 을 PIM 에 탑재해도 op-level token throughput 이 제한적.

## PULS 의 제안

*왜 attention 인가?* Attention 은 transformer 계열 모델에서 범용적으로 수행되는 연산이며 online streaming 으로 처리 가능 — PIM substrate 와의 자연스러운 정합.

- **HBM-PIM 아키텍처** — 위 한계를 회피하는 substrate 설계:
  - **Memory die 비침범 (P1)** — KV cache 가용 용량 잠식 회피
  - **Row-wise pipelined FSM (결정론적 cycle)** — thermal envelope 예측성 확보, dispatch 시점에 PIM 종료 시각 사전 계산
  - **Per-channel PIM / GPU 토글** — 별도 DMA 엔진 · bespoke 제어기 불요
  - **SP-PIM cross-GPU 2048 channel cooperation** — 단일 op-level 산출량 한계 분산 (8 GPU lock-step 협력)
  - **GQA / speculative attention 정합** — 모던 서빙 기능 호환
  - **Logic die SFU 내부 row-wise 누적** — logic die ↔ DRAM die 왕복 트래픽 회피
  - **Compute-bound timing 정합 (P5)** — TSV 경합 회피, PIM 활성화 구간을 GPU compute-bound op 실행 중에 한정
- **자체 스케줄러** — chunked-prefill + 혼합 배치 mixed batch primitive 와 호환 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §6):
  - **Event-driven dispatch + dependency DAG** — invariant (data dependency + resource) 을 그래프로 코드화, ready-node 선택 문제로 환원
  - **2-μ-batch lookahead** — 다음 μ-batch 작업을 미리 시작 + idle 자원을 다른 μ-batch 의 작업으로 채움 (자연 산출)
  - **풀 모델 구성 + 로컬 그리디 steering (노드 레벨)** — admission(풀 보충) ‖ decode-set steering ‖ prefill steering, 각각 고정 동작점 타깃을 독립 명중, 공정성을 위한 age-cap (전역 통계·idle-feedback 루프 없음)
  - **글로벌 스케줄러 (서버 레벨 노드 분배)** — 전체 도착 풀을 노드별로 라우팅: **그리디 콜드스타트**(긴 요청을 엣지노드로 shed + interleave-greedy 로 정상 노드를 평균 100K 로 충전) + **그리디 힐링**(완료마다 같은 크기로 per-completion 보충 = toxic-fit, 긴 요청 분포 보존·drift 0). **초반 ~2.2% 만 엣지로 보내면 이후 각 노드 풀이 평균 100K 동작점으로 유지**되어 로컬 steering 이 명중 — inter-node 이동·축출 0 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §7)
  - **축출 없는 결정론적 admission** — admit 시 full-length KV 를 예약하므로 일단 admit 된 요청은 *축출·recompute 되지 않음*(lost work 0). 메모리 압박은 preemption 이 아니라 admission backpressure(신규 거절)로 흡수하고, age-cap 이 대기를 상한(starvation 0). 공간(KV 캡)·시간(age-cap) 두 bound 가 직교 — 한 축의 압박을 다른 축의 회수로 갚지 않음

## 접근 요약

![PULS Instance Disaggregation](figures/instance_disaggregation.png)

- **Instance disaggregation** — 트랜스포머 레이어를 Instance A (attention block, 8 GPU, TP=8 + SP-PIM 2048 channel) 와 Instance B (post-attention block / FFN, 8 GPU, TP=8) 로 분리. 두 인스턴스는 inter-instance pipeline 으로 직렬 연결. KV cache 는 Instance A HBM 영구 보존 (인스턴스 간 KV 전송 없음). 본 분리가 산출하는 추가 효과:
  - **Instance B substrate cost 절감 가능성** — Instance B 는 FFN compute-bound 한정으로 HBM 의 full 대역폭이 B_cycle 에 기여하지 않음. **GDDR (또는 적은 stack 수 HBM)** 대체 시 unit cost 절감 가능 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §3.4 "Instance B Memory Substrate").
  - **고정 shape 텐서 handoff → straggler bubble 제거** — KV 길이 분산으로 decode batch 내 가변 shape 텐서가 발생하나, PIM 이 attention 단계에서 길이 의존성을 흡수하고 Instance B 에는 항상 고정 shape `[B × hidden]` 텐서만 전달. Instance B 는 ragged batching 없이 균일 GEMM 만 수행하여 decode batch 내 straggler bubble 제거 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §5.2).
  - **KV cache 이동 bus transaction 감소** — PIM 이 attention 을 in-place 처리하므로 GPU ↔ HBM 간 KV streaming traffic 자체가 제거됨. Long-ctx 영역에서 KV bytes read 가 attention 시간의 결정 요인이므로 효과 큼. **부수적 발열 감소 예상** (bus transaction energy ≫ compute energy) ([`ARCHITECTURE.md`](ARCHITECTURE.md) §5.7 보조 항목).
  - **FFN 가중치 bus transaction 감소** — PIM 이 attention 길이 의존성을 흡수하여 mixed batching 이 복원되면, prefill chunks 와 decode tokens 가 동일 FFN 가중치를 공유. Weight HBM traffic 의 token 분모가 확대되어 per-token FFN 가중치 bus transaction 감소 + arithmetic intensity 상승. **부수적 발열 감소 예상** ([`ARCHITECTURE.md`](ARCHITECTURE.md) §5.7 보조 항목).
- **HBM4 logic die SP-PIM**:
  - **Compute substrate** — row-wise pipelined attention SFU, 32-row tile FSM @ 1.3 GHz, FP16 MAC + FP8 (E4M3) KV cache (production FP8 KV 영역 정합)
  - **Channel-level PIM / GPU 토글** — HBM4 1 stack 당 32 channel 각자 PIM / 일반 모드 독립 토글
  - **SP-PIM cross-GPU 협력 (Q-replicate 병렬화)** — Instance A 8 GPU × 256 channel = 2048 channel 전체에서 단일 attention 을 lock-step 협력 처리. 동일 Q 벡터를 2048 channel 전체에 broadcast 하고 KV row 를 channel 에 sharding 하므로, 각 채널이 자기 KV slice 를 독립적으로 sweep → 단일 attention 연산이 2048 channel 단위로 병렬 실행
  - **Logic die SFU 내부 row-wise 누적** — attention 중간 결과 (softmax accumulator, row max) 가 SFU 내부에서 누적되어 logic die ↔ DRAM die 왕복 트래픽 회피
- **Interceptor host↔PIM 인터페이스** — decode-attention 단일 연산의 직접적 귀결 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §3.5):
  - 기존 JEDEC HBM4 RD/WR 명령의 **RFU (Reserved For Future use) bit 1개를 PIM_toggle 로 점유**하여 host↔PIM 인터페이스를 표준 DRAM 명령에 흡수. 별도 interrupt 불요.
  - FSM 결정론적 cycle 기반 **computed wait** 로 GPU 가 결과 read 시점 사전 계산.
  - PIM ↔ GPU 의 유일 채널은 HBM (PIM write → GPU read, GPU 내부 kernel 간 global memory 전달과 동일 패턴).

상세 — [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 워크로드 적용 범위

PULS 는 **특정 타겟 워크로드에 한정되지 않는다.** 배치 구성이 *길이분산 무관* steering 이기 때문이다:

- **Steering 은 풀의 평균 길이를 보지 않는다** — 짧은·긴 요청을 **조합**해 동작점 네 타깃(배포 128: decode 개수 62 ∧ Σkv 6.15M, prefill 128 토큰 ∧ depth-work 12.8M; OPERATING_POINT §4.1)만 맞춘다.
- 따라서 **decode 와 prefill 이 풀에 풍부한 어떤 분산-서버-스케일 워크로드든** — 길이 분포가 짧든 길든 혼합이든 bimodal 이든 — 동작점에 수렴한다.
- (균형 ctx ~100K 는 KV 캡 유도용 중간값일 뿐, 워크로드 강제값 아님.)
- **동작점(idle≈0) 도달 조건** = 풀에 decode·prefill 이 *풍부* (고동시성 분산 서빙의 대량 상주 decode 풀 + 지속 prefill). 실서버 정상상태가 정확히 이 영역. [Runtime Validation](#runtime-validation) 에서 **2 active μ-batch + 종속성·age-cap** 를 한 sim 에 넣고 동작점에 구성, **디코드 ≈99.5% 명중·Σdev ≈1.2% / 프리필 100%·Σdev ≈0.1%** 로 실증.
- **풀이 얕으면**(저부하·짧은 decode 만) PIM 또는 GPU-A 가 노는 건 *물리적 정상*(고칠 대상 아님) — 동작점은 풀이 풍부할 때의 균형점이다.

## 가속 Source 요약

| ID | Source | 영역 |
|---|---|---|
| F1 | SP-PIM attention (2048 channel Q-replicate, 8 GPU lock-step) | Op-level |
| F2 | Projection ‖ PIM attention double-buffering (intra-instance) | Op-level |
| F3 | Instance A–B inter-instance pipeline (steady-state) | Systems-level |
| F4 | μ-batch staggering (F2·F3 steady-state 전제) | Systems-level |
| F5 | Channel-independent PIM scheduling (KV 길이 분산 흡수) | Systems-level |
| (보조) | Mixed batching 복원 (가중치 공유 → arithmetic intensity ↑) | TTFT / throughput trade-off |
| (보조) | Bus traffic 절감 (PIM in-place attention → HBM-GPU bus transaction ↓) | Energy / cost |

## Mixed Batching 의 역할

- **1 차 목적: PIM TSV 점유 가능 시간 (compute-bound 헤드룸) 확보** — Pure decode 만으론 projection 이 memory-bound 로 전환되어 PIM 헤드룸 부족 가능. Prefill chunk 토큰을 mixed batch 에 추가 admit 하여 effective N 을 GEMM saturating 영역으로 유도, projection 을 compute-bound regime 에 유지.
- **부수 효과 (가중치 공유에서 파생)**:
  - prefill chunks 와 decode tokens 가 동일 가중치 공유 → arithmetic intensity 상승
  - per-token FFN 가중치 bus transaction 감소
  - inter-instance KV transfer 제거 (단일 instance 내 공존)

## Results

Llama-3 70B + DGX B200 + HBM4 substrate 위 4 가속 source (Aux1·Aux2·F3·F5) 의 calibrated projection 및 runtime 검증 (분산-서버 정상상태 합성 워크로드 위 통합 lifecycle sim).

> **시각 자료 동반** — source 별 도식 figure + 수치 분해 + runtime validation + 정직한 disclosure 를 interactive reading guide 의 **§6 Acceleration Sources** · **§7 Results** 로 정리: [`docs/scheduler_policy.html`](docs/scheduler_policy.html) ([online](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html#sec-7)).

### Substrate

| 항목 | 값 |
|---|---|
| GPU compute | DGX B200 standalone, GPU 당 FP16 dense 2,200 TFLOPS |
| Memory | HBM4 hypothetical projection, GPU 당 16 TB/s × 8 GPU = 128 TB/s aggregate |
| η_HBM_external | 0.74 (fix) + sensitivity sweep {0.70, 0.74, 0.80} |
| PIM tile time | 267 ns (compute-bound, FP8 KV) |
| RTL FSM | 1.3 GHz · 347 cycles |
| Model | Llama-3 70B (L=80, hidden=8192, FFN intermediate=28672) |

### Per-source Acceleration

| Source | Mechanism | Result |
|---|---|---|
| Aux1 | Mixed batching 가중치 재사용 | **2.0×** (closed-form), 1.97× (Colab T4 측정) |
| Aux2 | KV bus traffic 감소 | **4.95× speedup, 79.8% 감소** |
| F3 | Inter-instance pipeline ratio | 0.92–0.99 (closed-form ctx sweep); 통합 lifecycle sim 이 **2 active μ-batch 를 동작점에 구성, 디코드 ≈99.5%·Σdev ≈1.2% / 프리필 100%·Σdev ≈0.1% (balance 발현)** — [Runtime Validation](#runtime-validation) |
| F5 | Channel-independent vs lock-step | **5.15× speedup** (KV variance dominant) |

### Aggregate Speedup

| 항목 | Baseline | PULS | Saving |
|---|---|---|---|
| Weight streaming | 2.89 ms | 1.45 ms | 1.45 ms |
| KV bus traffic | 8.25 ms | 1.67 ms | 6.59 ms |
| **합계 (weight + bus)** | **11.14 ms** | **3.12 ms** | **8.04 ms (72.2%)** |

Net speedup: **3.57× (closed-form, weight + bus)** → **4–5× (F5 포함)**.

### Runtime Validation

**통합 lifecycle 시뮬레이션 (PULS 독립, composition 명중).** 단일 실 trace 대신 분산-서버 정상상태를 대표하는 합성 워크로드를 *고정* 해두고, 콜드스타트 → 운영(steering · prefill→decode 전이 · per-completion 힐링 · age-cap)을 한 sim 에 넣어 동작점 composition 이 유지되는지 본다 — 답을 정해놓고 seed 를 튜닝하지 않음.

- **워크로드(합성)** — wide·다양 길이 풀(1K\~1M, short/mid/long 혼합), prefill·decode 풍부. **warm-start** = 정상상태 스냅샷(각 요청을 생애 랜덤 지점에 배치, cold-start 램프 생략).
- **모델 — 2 active μ-batch (3 아님).**
  - 한 노드는 2 μ-batch 만 동시 active(F2/F3 overlap).
  - 한 배치의 forward pass 가 끝나면 그 멤버가 풀로 돌아오고 **(반환분 + 상주 잉여)에서 다시 1 배치 재선택**(메모리 할당 0) — *3번째 배치를 강제 구성하지 않는다.*
  - 완료 요청은 per-completion 힐링으로 같은 크기 보충, prefill 완료는 decode 로 전이, 공정성 age-cap = 5. 배포 prefill 128.

**composition — 동작점 명중 ([puls-engine/sim/lifecycle.cpp](puls-engine/sim/lifecycle.cpp), 종속성·age-cap 포함; 재현: `puls_lifecycle 4000 64 5 2000`):**

| 2 active μ-batch (완료시 재구성) | 동작점 타깃 | 명중 | Σdev |
|---|---|---|---|
| **decode** | 62 ∧ Σkv 6.15M | **≈99.5%** | **≈1.2%** |
| **prefill** | 128 토큰 ∧ depth-work 12.8M | **100%** | **≈0.1%** |

> **(2026-06 sim 충실화 정정)** 이전 디코드 100%/Σdev 0.38% 는 lifecycle sim 의 힐링
> 버그(센터링 admit `ideal≈ctx_balance` → 풀 all-mid 붕괴, 긴 요청 0% → composition trivial)에서
> 나온 값이었다. canonical 힐링(per-completion `ideal=hole`, like-for-like) + 엣지 게이팅 +
> prompt-무관 현실 decode 길이 + best-of-2000 무한풀 근사로 정정하면, 분포 보존(≈20/70/10)된
> 다양 풀에서 디코드 ≈99.5%/Σdev≈1.2% (age_cap 5) — §3 sweep 의 cap5 spread(0.7%)와 정합.
> 프리필 100%/Σdev≈0.1%.

해석:

- **age-cap 꼬리 없음 (2-active 구조).** 옛 검증의 "3번째 배치 튐(108개·spread 3.7%)"은 *강제된 3rd 배치*에서 warm-start 대기 멤버가 age-cap 을 유발한 것. 2-active 모델은 3rd 를 강제하지 않고 (완료분 + 잉여)로 *재구성*만 하므로 그 꼬리가 구조적으로 사라진다.
- **종속성·age-cap 넣고도 유지.** prefill→decode 전이와 공정성 age-cap = 5 를 다 넣어도 두 composition 이 무너지지 않음 — 로직(steering · greedy · healing · age-cap · KV-센터링)은 *스케일 불변*, 동작점만 상수(prefill 256↔128 동형).
- **throughput 은 설계상 지속.**
  - per-cycle decode 예산이 동작점에 고정(62, KV 캡에 먼저 닿으면 그 이하)이라 풀이 풍부한 한 매 cycle 이 고정 토큰 양을 처리 → 지속성은 동작점 고정의 구조적 귀결.
  - 절대 tok/s 는 silicon 보정 cycle 시간 필요.
- **클러스터 스케일 — 글로벌 스케줄러.**
  - 서버스케일(노드 수백–수천)에선 글로벌 도착 평균이 100K 보다 높아 노드별 풀이 drift → idle 폭발.
  - **글로벌 스케줄러**(greedy 콜드스타트: 긴 것 엣지 shed + interleave-greedy · per-completion 힐링 = toxic-fit)로 **초반 ~2.2% 엣지 비용만 감수하면 그 뒤로 각 노드를 평균 100K 동작점에 무한정 유지**(긴 요청 보존, drift 0).
  - 원리·E 스윕·on2 측정은 [`ARCHITECTURE.md`](ARCHITECTURE.md) §7 / [puls-engine/core/global_scheduler.cpp](puls-engine/core/global_scheduler.cpp).

<a id="cluster-scheduler-c"></a>

## 클러스터 스케줄러 (C)

승리한 클러스터 설계 **C** 는 §6 노드 메커니즘(상주 **잉여(surplus)** + iteration 마다 steering 재선택 + per-completion 힐링)을 유지하면서, 클러스터 레벨 두 조각을 추가한다 — **글로벌 age-cap**(노드 간 FIFO 공정성 → 강제 라우팅)과 **on-node 멀티턴 KV 캐시**(3-tier HBM / SSD / recompute, decode 풀을 쓰고 남은 HBM 에 배치).

- **셋이 역할을 나눠 맡는다** — 잉여 → composition(Σdev), 글로벌 age-cap → 지연/공정성(그리고 돌아온 세션을 캐시에 명중할 만큼 빠르게 서빙해주는 것도 이쪽), 캐시 → TTFT.
- **잉여 재선택은 또한 임의의 배치 교란의 *범용 흡수기*임이 입증됐다** — age-cap 강제 주입이든 어피니티 길이 불일치든 — 그 덕에 어피니티를 새 기계장치 0 으로 추가할 수 있었다.
- **동일한 TBT 가 이제 인스턴스 종속성 모델을 담는다** — Instance A (PIM ‖ GPU-A) 가 Instance B (FFN) 보다 먼저 끝나야 하고, PIM↔GPU-A 가 HBM 을 공유하므로 `TBT = max(instance_a, t_ffn) × layers`, 여기서 `instance_a = max(t_pim, t_gpu_a) + β·max(0, t_pim − t_gpu_a)` (`t_pim ≤ t_gpu_a` 일 때 경합 없음).

> **두 모델링 축 — 혼동 금지.** `max_tokens` 는 **랜덤 EOS 출현**을 모델링한 것 — 각 요청의 시간 축 위 decode 루프 길이(EOS 까지 몇 토큰인가, 랜덤 EOS 를 샘플된 결정론적 상수로 치환)이지 KV 양이 *아니다*. **KV**(`live_kv = prompt + dec`, ≈6.2M)는 매 step *레이어마다* 읽는 양 — PIM 이 레이어별 attention 에서 읽으므로 `TBT = (레이어당 max) × 80 layers`.

**캐시 어피니티 (멀티턴 복귀).** KV 가 HBM 에 캐시된 복귀는 보유 노드의 대기열에 줄을 서고 hole 발생 시 1순위로 admit 된다 — 명중의 **99.7%** 가 물리적이 됨 (옛 회계는 아무 노드 명중이나 집계; 물리적으로는 1.7% ≈ 1/Z 만 보유 노드에 안착).

```
노드 z에서 hole 발생
  1순위: z의 어피니티 대기열 비어있지 않음 → 가장 오래 기다린 대기자 admit (길이·cap_room 무관)
  2순위: 대기열 비었으면 → 평소처럼 queue.pull_slot(hole)
          (글로벌 aged 있으면 강제, 없으면 ideal=hole 최근접)
```

작동 이유:

- hole 은 노드당 ~17 라운드마다 도착 — 대기는 짧고 자연적으로 상한이 있다.
- 대기 비용 ~2 ms/round vs 캐시 포기 비용 = SSD reload 160–780 라운드 (~10× 이상) — 그래서 별도 spill cap(200, 글로벌 25 와 별개)이 캐시된 복귀를 기다리게 한다. 실측 spill 은 사실상 0: **어피니티 후보 13,483 건 중 4 건(0.03%)** — 글로벌 cap 25 를 그대로 쓰던 설계에선 31%(4,317 건)였고, 전용 cap 분리가 이를 ~1,000× 줄였다.
- 어피니티 복귀는 구조적으로 거의 like-for-like (그 노드의 옛 거주자, growth ≤12K, eligibility ≥16K 가 고성장 short 를 필터) — 분포가 거의 보존된다.
- 잔여 불일치는 기존 잉여 + 재선택 + 노드 age-cap 기계장치가 흡수 — Σdev 는 spill 대비 오히려 개선 (spill → 글로벌 강제 주입이 더 나쁜 교란).

배포 동작점에서 측정 ([puls-engine/sim/csched.cpp](puls-engine/sim/csched.cpp), 8000 iters · Z = 64 · `csched 8000 64 16000 200 25 300 0.5 1e8 5 25`):

| config (잠금) | 값 | | KPI (C, 실측) | 값 |
|---|---|---|---|---|
| decode_surplus | **25** (decode_pool 149) | | Σdev (avg / worst) | **1.38% / 22.6%** |
| global_age_cap | **25** | | TBT (mean / p99) | **2058 / 2194 µs** |
| aff_spill | **200** | | TTFT (mean / p99) | **0.76M / 8.77M µs** |
| affinity / dyncache | **on** | | SLO goodput | **3.80M tok/s** |
| eligibility | **16000** (mid · long) | | TTFT-met | **97.2%** |
| evict_age | 200 | | cache HBM-hit | **91.1% (물리적 99.7%)** |
| contention β | 0.5 (보수적; C 는 β-robust) | | max wait | **24 rounds** |
| offload_bw | 1e8 (SSD ≈ 50 GB/s NVMe array) | | worst token gap | **6 rounds** (= 노드 age-cap+1) |
| node_age_cap | 5 | | | |

- **[PAUSE] KPI — per-request 토큰 간격.** 배치 레벨 TBT 에는 보이지 않는다 (노드-cap 스윕 전체에서 TBT p99 동일); 노드 age-cap 이 최악 간격을 정확히 cap+1 라운드로 상한 (cap 5 → 12.2 ms, 일시정지 토큰 3.4%, 평균 간격 1.12) — 순수 tail-bound knob.

- **C 는 두 대안을 모든 실 KPI 에서 능가한다** — 노드-로컬 baseline A(글로벌 age-cap 없음 → TTFT 0.94M · `max_wait` 4551, 무상한 starvation, 길이 편향; 캐시 off)와 순수 pre-positioning B(잉여 없음 → Σdev 20.1%, PIM exposed 82.5%, goodput 2.15M tok/s).
- **현실적 SSD 상수에서 A 는 상당히 회복하므로**(TTFT 1.75M → 0.94M), C 의 남은 우위는 상한 있는 대기(24 vs 4551)·composition·물리적으로 실재하는 캐시 tier 에 있다 — 더 작지만 더 방어 가능한 우위.
- **도출된 동작점은 그대로**(62 · 6.15M · ~100K · prefill 128) — 잉여 / age-cap / 캐시 / β 는 그 *위에서* 스윕하는 클러스터-레이어 knob 이지 동작점 변경이 아니다.
- 전체 A/B/C 비교·스윕·경합 분석은 [`ARCHITECTURE.md`](ARCHITECTURE.md) §7.6.

## Limitations / Disclosure

- **Hardware 미보유** — 실제 H100 / HBM4 silicon 없음. 상대적 source decomposition 은 calibrated ([Results](#results) 참조).
- **HBM4 substrate** — hypothetical projection (현재 production 부재; ARCH §3.1 literal 정합). JEDEC JESD270-4A spec + 자체 Ramulator2 cycle-accurate 측정 (FP8 / FP16 tile load · PIM compute) 인용.
- **η_HBM_external** — H100 HBM3 측정값을 HBM4 로 확장 (Framing A).
- **RTL substrate** — open-source flow (Yosys + ASAP7 + OpenSTA pre-CTS) 한정. Commercial signoff 영역 외.
- **Runtime Validation = 합성 워크로드** — 분산-서버 정상상태(대량 상주 decode 풀 + 지속 prefill)를 대표하는 합성 분포 + warm-start seed (cold-start 램프 생략). composition(per-cycle 균형)이 검증 대상.
- **비교 baseline (vLLM / Sarathi-Serve) + F1·F2 ablation** — 후속 calibration 으로 연기.
- **단일 vendor production trace** — 공개 long-ctx agentic production trace 가 사실상 1 종 한정. 1M-class benchmark dataset + mid-ctx production chat trace 를 보강 axis 로 사용.
- **Main claim 정량 = projection** — Pre-silicon 정량 수치는 *추정* + provenance label 동반 ([Results](#results) 참조).

## Interactive Reading Guide

PULS 의 핵심 스케줄러 정책 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §6) 은 interactive single-page companion 으로도 제공된다. 가이드는 다섯 개의 navigable 섹션 — instance disaggregation, invariants & dispatch DAG, scheduler 사용법 (pseudo-code), adaptive admission, worked dispatch trace 예시 — 으로 구성되며, 순서대로 읽거나 주제별로 점프해 읽을 수 있다.

- **Online (rendered HTML, GitHub Pages 호스팅):** [rhydcdc.github.io/puls-rfc/scheduler_policy.html](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html)
- **Local:** repo 를 clone 하고 [`docs/scheduler_policy.html`](docs/scheduler_policy.html) 을 브라우저에서 열기

## Repository

- [`README.md`](README.md) — 본 문서 (entry point)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 아키텍처 본문 (motivation, design principles, substrate, instance disaggregation, scheduler integration, 풀 모델 admission + 2-active 구성 검증 §6.8, layer flow, prior art 비교)
- [`OPERATING_POINT.md`](OPERATING_POINT.md) — Phase-2 동작점 & 배치 구성 canonical spec (풀 모델, steering 타깃, 동작점 구성 근거)
- [`puls-engine/`](puls-engine/CONTRACT.md) — 모델·HW 일반화 C++ 스케줄러 (derive · steering · node/global scheduler · lifecycle sim, 196 checks)
- [`docs/scheduler_policy.html`](docs/scheduler_policy.html) — interactive reading guide (ARCHITECTURE.md §6 동반)
- [`LICENSE`](LICENSE) — Apache 2.0

빌드·검증: `cd puls-engine && bash build.sh` → 196 checks.

## License

Apache License 2.0. 상세 = [`LICENSE`](LICENSE).
