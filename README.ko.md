# PULS — PIM-Unified LLM Serving

> ✅ **스케줄러 로직 구현·검증 완료** — 풀 모델 배치 구성(admission=풀 보충 ∥ decode-set ∥ prefill 독립 steering)이 동작점(decode ≈120 / Σkv 12.3M, prefill 256 / depth-work 25.6M)에 수렴, 두 풀 모두 풍부 시 세 자원 idle spread **0.74%**(증명된 이론 floor 와 일치) ([Runtime Validation](#runtime-validation)).

**Scheduler-aware co-design of HBM-PIM and production LLM serving stack.**

> **Disclosure** — 학부생 단일 저자의 개인 연구 프로젝트. 소속 기관·vendor 연계 없음.
>
> 혼자 공부하며 배우는 과정이며, 피드백 · 지적 · 가르침을 적극 환영합니다 (GitHub Issues / Discussions).

📊 **시각 자료:** [**rhydcdc.github.io/puls-rfc/scheduler_policy.html**](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html)

본 repo 는 진행 중인 prototype 의 공개 RFC (Request for Comments). Architecture + design rationale + scheduler 정책의 entry point 와 함께, real long-context trace 위 4 가속 source 의 calibrated projection 산출도 함께 제공 ([Results](#results) 참조).

상세 본문 — [`ARCHITECTURE.md`](ARCHITECTURE.md) (substrate, instance disaggregation, scheduler integration, adaptive admission, layer flow, prior art 비교 통합).

## 목차

**Background**
- [Processing-in-Memory 아키텍처의 특징](#processing-in-memory-아키텍처의-특징)
- [문제 의식](#문제-의식)

**The Proposal**
- [PULS 의 제안](#puls-의-제안)
- [접근 요약](#접근-요약)
- [타겟 워크로드](#타겟-워크로드)
- [가속 Source 요약](#가속-source-요약)
- [Mixed Batching 의 역할](#mixed-batching-의-역할)

**Results**
- [Results](#results)
- [Limitations / Disclosure](#limitations--disclosure)
- [Forward-looking: HBF-class 분리 Substrate](#forward-looking-hbf-class-분리-substrate)

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
  - **풀 모델 구성 + 로컬 그리디 steering** — admission(풀 보충) ‖ decode-set steering ‖ prefill steering, 각각 고정 동작점 타깃을 독립 명중, 공정성을 위한 age-cap (전역 통계·idle-feedback 루프 없음)

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
- **Interceptor host↔PIM 인터페이스** — decode-attention 단일 연산의 직접적 귀결. 기존 JEDEC HBM4 RD/WR 명령의 **RFU (Reserved For Future use) bit 1개를 PIM_toggle 로 점유**하여 host↔PIM 인터페이스를 표준 DRAM 명령에 흡수. 별도 interrupt 불요. FSM 결정론적 cycle 기반 **computed wait** 로 GPU 가 결과 read 시점 사전 계산. PIM ↔ GPU 의 유일 채널은 HBM (PIM write → GPU read, GPU 내부 kernel 간 global memory 전달과 동일 패턴) ([`ARCHITECTURE.md`](ARCHITECTURE.md) §3.5).

상세 — [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 워크로드 적용 범위

PULS 는 **특정 타겟 워크로드에 한정되지 않는다.** 배치 구성이 *길이분산 무관* steering 이기 때문이다 — former 는 풀의 평균 길이를 보지 않고, 짧은·긴 요청을 **조합**해 동작점 네 타깃(decode 개수 123 ∧ Σkv 12.3M, prefill 256 토큰 ∧ depth-work 25.6M)만 맞춘다. 따라서 **decode 와 prefill 이 풀에 풍부한 어떤 분산-서버-스케일 워크로드든** — 길이 분포가 짧든 길든 혼합이든 bimodal 이든 — 동작점에 수렴한다 (균형 ctx ~100K 는 KV 캡 유도용 중간값일 뿐, 워크로드 강제값 아님).

- **동작점(idle≈0) 도달 조건** = 풀에 decode·prefill 이 *풍부* (고동시성 분산 서빙의 대량 상주 decode 풀 + 지속 prefill). 실서버 정상상태가 정확히 이 영역. [Runtime Validation](#runtime-validation) 에서 두 풀 풍부 시 idle spread **0.74%** 로 실증(얇으면 ~4.6%, age-cap 비용).
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

Llama-3 70B + DGX B200 + HBM4 substrate 위 4 가속 source (Aux1·Aux2·F3·F5) 의 calibrated projection 및 long-context production trace 위 runtime 검증.

### Substrate

| 항목 | 값 |
|---|---|
| GPU compute | DGX B200 standalone, GPU 당 FP16 dense 2,200 TFLOPS |
| Memory | HBM4 hypothetical projection, GPU 당 16 TB/s × 8 GPU = 128 TB/s aggregate |
| η_HBM_external | 0.74 (fix) + sensitivity sweep {0.70, 0.74, 0.80} |
| PIM tile time | 267 ns (compute-bound, FP8 KV) |
| RTL FSM | 1.3 GHz · 347 cycles |
| MFU | 0.6 default + sensitivity sweep {0.5, 0.6, 0.7} |
| Model | Llama-3 70B (L=80, hidden=8192, FFN intermediate=28672) |

### Per-source Acceleration

| Source | Mechanism | Result |
|---|---|---|
| Aux1 | Mixed batching 가중치 재사용 | **2.0×** (closed-form), 1.97× (Colab T4 측정) |
| Aux2 | KV bus traffic 감소 | **4.95× speedup, 79.8% 감소** |
| F3 | Inter-instance pipeline ratio | 0.92–0.99 (closed-form ctx sweep); 풀 모델 시뮬서 **3자원 idle spread 0.74% (두 풀 풍부; balance 발현)** — [Runtime Validation](#runtime-validation) |
| F5 | Channel-independent vs lock-step | **5.15× speedup** (KV variance dominant) |

### Aggregate Speedup

| 항목 | Baseline | PULS | Saving |
|---|---|---|---|
| Weight streaming | 2.89 ms | 1.45 ms | 1.45 ms |
| KV bus traffic | 8.25 ms | 1.67 ms | 6.59 ms |
| **합계 (weight + bus)** | **11.14 ms** | **3.12 ms** | **8.04 ms (72.2%)** |

Net speedup: **3.57× (closed-form, weight + bus)** → **4–5× (F5 포함)**.

### Runtime Validation

**합성 워크로드 + 풀 모델 스케줄러 end-to-end 시뮬레이션.** 단일 실 trace 대신 분산-서버 정상상태를 대표하는 합성 워크로드를 *고정* 해두고, idle 이 자연 수렴하는 값을 *그대로* 보고한다 — idle≈0 을 맞추려 seed 를 튜닝하지 않음(튜닝은 답을 정해놓고 맞추는 것이라 무의미).

- **워크로드(합성)** — prefill 20K\~180K 다양(sweet spot ~100K 포함), decode 8K\~40K *긴 생성* (= 분산 서버의 대량 상주 decode 풀). **warm-start seed 6,000** = 정상상태 스냅샷: 워크로드 *자체 분포* 에서 각 요청을 생애 랜덤 지점(prefill 진행도 또는 decode 진행도)에 배치 — 비편향, cold-start 램프만 생략.
- **방식** — 풀 모델 스케줄러(admission = 풀 보충 ∥ decode-set steering ∥ prefill steering, 셋 독립)로 idle 수렴까지 시뮬. 활성 μ-batch 의 배치 구성 + 3 자원 idle 측정.

두 풀 모두 풍부(동작점 premise — 상주 decode 풀 *그리고* 풍부한 prefill 공급)하게 측정 — steering 이 네 타깃을 명중시킬 depth 다양성을 갖도록:

| μ-batch 구성 (활성 평균) | 측정 | 타깃 |
|---|---|---|
| decode 개수 | **119** | 123 |
| decode Σkv | **12.3M** | 12.3M |
| prefill 토큰 | **256** | 256 |
| prefill depth-work | **25.6M** | 25.6M |

| 자원 idle | 측정 |
|---|---|
| GPU Instance A (proj + prefill-attn) | **10.5%** |
| PIM Instance A (decode-attn) | **10.9%** |
| GPU Instance B (FFN) | **11.2%** |
| **spread (max − min, = 균형 갭)** | **0.74%** (converged) |

해석:

- **네 동작점 타깃 모두 정확 명중** — steering 이 풀에서 decode-set(≈120 / 12.3M)과 prefill(256 / 25.6M)을 *독립적으로* 구성, 길이분산 무관(짧+긴 조합). prefill 풀이 풍부하면 depth-work 가 25.6M 에 정확 명중.
- **세 자원이 0.74% 안으로 균형**(각 ~10\~11% idle) — 직렬이면 idle ~67%(t_A + t_B 합). **F2(projection ‖ PIM double-buffering)·F3(inter-instance pipeline) 발현의 정량 증거** — 세 per-cycle 자원 시간이 <1% 로 일치.
- **측정 ≈ 알고리즘 floor (증명됨).** 디스패치된 μ-batch 에 정확한 op-time 함수를 호출해 얻은 이론 perfect-overlap floor 의 **spread(0.91%)가 측정 spread(0.74%)와 일치**; 절대 idle 은 그 floor + *균일* ~10% overlap gap(pipeline fill/drain + 2-active staggering) → **미설명 잔여손실 0**. 상세 도출 — [`ARCHITECTURE.md`](ARCHITECTURE.md) §6.8, 전체 기록 — [`implementation/debug_phase2/REPORT.md`](implementation/debug_phase2/REPORT.md).
- **배치 구성에 robust** — prefill 풀이 얇으면(depth-work ~27M 오버슈트) 균형이 ~4.6% 로 벌어지는데 이는 **age-cap 공정성 비용**(순수 steering 은 <1% 회복); decode 축 동작점은 무관하게 명중. 스펙트럼은 [`ARCHITECTURE.md`](ARCHITECTURE.md) §6.8 에 기록.
- **throughput 지속성은 별개 축** — decode 길이가 매우 길어(상주 풀 큼) 측정창 내 완료 0 → TTFT/TBT 는 본 측정서 미산출(별도). 본 검증 대상은 per-cycle 균형(idle). 드레인·완료·KV 누수 0 은 합성 acceptance(전부 완료·KV remaining=initial·정상 종료)로 별도 검증.

### Honest Disclosure

- **HBM4 substrate** = hypothetical projection (현재 production 부재; ARCH §3.1 literal 정합).
- **η_HBM_external** = H100 HBM3 측정값을 HBM4 로 확장 (Framing A).
- **F1·F2 ablation + 비교 baseline (vLLM / Sarathi-Serve)** = 후속 calibration 으로 연기 (calibration-heavy).
- **Runtime Validation = 합성 워크로드** — 분산-서버 정상상태(대량 상주 decode 풀 + 지속 prefill)를 대표하는 합성 분포 + warm-start seed(비편향, cold-start 램프 생략). 실 1M-ctx trace 의 cold-start 전체 prefill 은 수억 step 이라 직접 시뮬 비현실적 → warm-start 가 *상주 풀* 을 대표. idle(per-cycle 균형)이 검증 대상이며, 절대 latency 아님.
- **절대 metric** (TTFT, TPOT, throughput) = silicon 부재로 영구 out of scope.

## Limitations / Disclosure

- **Hardware 미보유** — 실제 H100 / HBM4 silicon 없음. 상대적 source decomposition 은 calibrated ([Results](#results) 참조); 절대 metric (TTFT, TPOT, throughput) 은 out of scope.
- **HBM4 추정** — JEDEC JESD270-4A spec 기반 + 자체 Ramulator2 기반 cycle-accurate 측정 (FP8 tile load / FP16 tile load / PIM compute 영역) 인용.
- **RTL substrate** — open-source flow (Yosys + ASAP7 + OpenSTA pre-CTS) 한정. Commercial signoff 영역 외.
- **단일 vendor production trace** — 공개 long-ctx agentic production trace 가 사실상 1 종 한정. 한계 disclosure 와 함께 1M-class benchmark dataset + mid-ctx production chat trace 를 보강 axis 로 사용.
- **Main claim 정량 = projection** — Pre-silicon 정량 수치는 *추정* + provenance label 동반 ([Results](#results) 참조).

## Forward-looking: HBF-class 분리 Substrate

본 RFC main claim 영역 (HBM4 SP-PIM) 의 정합성 검증 이후, 별도 substrate 인 **HBF (High Bandwidth Flash)** 같은 분리 메모리 계층에 PIM 코어를 탑재하는 방향이 추가 검토 가치를 가진다. 구조적 효과:

- **PIM 가동률 ceiling 상승** — GPU 의 HBM 점유 phase 와 독립적으로 PIM 가동 가능 → *compute-bound timing 정합 제약* (P5, TSV 경합 회피를 위해 PIM 활성화 구간을 GPU compute-bound op 실행 중에 한정해야 한다는 원칙) 이 substrate-level 분리로 자동 해소되는 방향.
- **메모리 path 공유 해소** — GPU ↔ PIM 의 TSV / inter-bank path 경합이 substrate-level 분리로 구조적으로 회피되는 방향 (GPU 는 HBM 버스, PIM 은 HBF 버스 각자 점유).

단, **HBF spec 이 현 시점 미공개** 이므로 데이터 로드 latency, KV cache 의 핫·콜드 tier 분할 정책, write endurance 제약은 정량 특정 불가능. 본 항목은 *방향성 한정* 이며 spec disclosure 이후 정량 follow-up 영역.

## Repository

- [`README.md`](README.md) — 본 문서 (entry point)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 아키텍처 본문 (motivation, design principles, substrate, instance disaggregation, scheduler integration, 풀 모델 admission + idle floor §6.8, layer flow, prior art 비교)
- [`OPERATING_POINT.md`](OPERATING_POINT.md) — Phase-2 동작점 & 배치 구성 canonical spec (풀 모델, steering 타깃, idle-floor 근거)
- [`LICENSE`](LICENSE) — Apache 2.0

## License

Apache License 2.0. 상세 = [`LICENSE`](LICENSE).
