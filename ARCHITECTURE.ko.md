# PULS Architecture

**P**IM-**U**nified **L**LM **S**erving — scheduler-aware co-design.

- Motivation / problem statement / proposal 개관 — [`README.md`](README.md) 참조
- 정량 평가 (가속 배수, latency / throughput 절대값) — Phase 3 calibration 영역에서 측정 예정

## Table of Contents

- [1. Key Observations](#1-key-observations)
- [2. Design Principles](#2-design-principles)
- [3. Architecture](#3-architecture)
  - [3.1 Compute Substrate](#31-compute-substrate)
  - [3.2 Channel-level PIM Toggle](#32-channel-level-pim-toggle)
  - [3.3 KV Cache Placement](#33-kv-cache-placement)
  - [3.4 Instance Disaggregation: Attention Block vs FFN Block](#34-instance-disaggregation-attention-block-vs-ffn-block)
  - [3.5 Host↔PIM Interface (Interceptor)](#35-hostpim-interface-interceptor)
- [4. Op Partitioning](#4-op-partitioning)
- [5. Scheduler Integration](#5-scheduler-integration)
  - [5.1 Phase-aware Channel Split](#51-phase-aware-channel-split)
  - [5.2 Fixed-shape Handoff to Instance B](#52-fixed-shape-handoff-to-instance-b)
  - [5.3 Compute-bound 구간 중 PIM 연산 Overlap](#53-compute-bound-구간-중-pim-연산-overlap)
  - [5.4 스케줄링 예측 가능성의 부분적 해소](#54-스케줄링-예측-가능성의-부분적-해소)
  - [5.5 Prototype Vehicle: Self-authored Scheduler Framework](#55-prototype-vehicle-self-authored-scheduler-framework)
  - [5.6 Intra-instance Double-Buffering](#56-intra-instance-double-buffering)
  - [5.7 가속 Source 분해](#57-가속-source-분해)
- [6. Instance A Scheduler 내부 정책](#6-instance-a-scheduler-내부-정책)
  - [6.1 μ-batch 구성](#61-μ-batch-구성)
  - [6.2 Invariants](#62-invariants)
  - [6.3 Dispatch Policy: Event-driven + Dependency DAG](#63-dispatch-policy-event-driven--dependency-dag)
  - [6.4 Adaptive Admission](#64-adaptive-admission)
  - [6.5 Example Dispatch Trace](#65-example-dispatch-trace)
  - [6.6 Bound 분석](#66-bound-분석)
  - [6.7 구현 요건](#67-구현-요건)
- [7. Orthogonality to Complementary Techniques](#7-orthogonality-to-complementary-techniques)
  - [7.1 Paged KV Memory Management](#71-paged-kv-memory-management)
  - [7.2 Speculative Attention](#72-speculative-attention)
  - [7.3 Prefix KV Caching](#73-prefix-kv-caching)
- [8. Open Empirical Work](#8-open-empirical-work)

---

## 1. Key Observations

**O1.** Continuous-batched decode 에서 요청별 KV 길이의 분산은 attention 연산 시간의 분산을 유발하며, 이는 GPU 배치 단위에서 straggler bubble 로 나타난다.

**O2.** Attention 이후의 모든 연산 (Out projection, FFN) 은 요청 간 동일한 가중치와 동일한 텐서 형상을 사용한다. 즉 가변 길이 의존성은 오직 attention 단계에 국한된다.

**O3.** GPU 의 HBM 대역폭 사용률은 phase 에 따라 큰 분산을 보인다. Prefill 의 FFN 및 projection 은 tensor core compute-bound 상태에서 HBM 대역폭의 상당 부분이 idle 상태로 존재한다.

## 2. Design Principles

**P1. GPU 자원 비침범.** PIM 은 GPU compute 자원을 점유하거나 stall 시키지 않는다.

**P2. Memory-bound 연산만 담당.** Arithmetic intensity 가 낮고 weight / activation streaming bound 인 연산 (decode attention) 만 PIM 이 담당한다. Compute-bound 연산 (projection, FFN, prefill attention) 의 PIM 탑재는 logic die / DRAM die 두 substrate 모두에서 불성립:

- **Logic die 배치 한계** — Logic die 의 *면적 · thermal envelope · SRAM 부재의 삼중 제약* 으로 PIM compute density 가 GPU tensor core 대비 한 자릿수% 수준에 머물러, 가속 기여가 미미.
- **DRAM die 배치 한계** — (i) MAC 영역이 메모리 cell 을 잠식하여 KV cache 가용 용량 감소, (ii) DRAM 의 thermal sensitivity 로 sustained 운용 시 throttling, (iii) 비표준 DRAM 회로 수정 필요 → fab risk · 수율 부담.

따라서 PULS 는 compute-bound 연산을 GPU tensor core 에 잔존시키고, PIM 은 memory-bound 영역에만 한정한다.

**P3. 가변 길이 흡수.** 요청 간 길이 분산을 유발하는 attention 을 PIM 이 처리하여, GPU 에는 항상 고정 형상의 텐서가 전달되도록 한다.

**P4. 스케줄러 가시성.** PIM 가용 채널 수와 동작 phase 는 서빙 스케줄러가 제어 가능한 노출된 dial 이다.

**P5. TSV 점유와 Compute-bound Timing 정합.**

- **TSV / row buffer 점유로 외부 버스 경합** — PIM 활성화 채널은 TSV · row buffer 를 점유하므로 같은 stack 의 GPU-mode 채널과 외부 HBM 버스를 경합.
- **Compute-bound timing 활성화** — PIM 은 GPU 가 compute-bound 영역에 진입한 timing (QKV / O projection 또는 FFN 실행 중) 에 활성화되어야 외부 BW 경합 회피.

P2 (memory-bound 연산만 담당) ∩ P5 (compute-bound timing 활성화) = PULS 의 PIM dispatch 정책 — *memory-bound 연산을 처리하되, compute-bound 연산이 진행 중인 timing window 에 한정하여 동작.* §5.1 phase-aware channel split, §5.3 overlap policy 의 design rationale.

## 3. Architecture

### 3.1 Compute Substrate

HBM4 **logic die (PHY)** 에 row-wise pipelined attention SFU 를 배치한다 — 메모리 die 비침범. 사양:

- Head dim 128. **FP16 MAC 코어 / FP8 (E4M3) KV-cache 저장.** 가중치 및 activation 은 BF16/FP16 유지.
- **32-row tile 단위 FSM, 1.3 GHz clock.**
  - Tile data flow — SFU 가 KV tile 을 FP8 로 HBM 에서 읽어 per-tile dequant 후 FP16 MAC 실행.
  - **FP8 KV regime** — tile TSV load 시간 < PIM compute 시간 → tile 실행이 *compute-bound* regime, FSM 의 deterministic timing 성립.
  - **FP16 KV regime** — 동일 매핑에서 tile 이 load-bound 로 전환, 약 2× 시간.
  - Regime 분기는 cycle-accurate 측정으로 확정 (Ramulator2 기반).
- **GEMM / GEMV 통합 처리** — 행 단위 pipelined FSM 이므로 column 폭이 1 (GEMV, decode B=1) 이든 small matrix (GEMM, batch decode · multi-head) 이든 동일 control flow. Per-tile MAC width 만 다를 뿐 FSM cycle 구조 불변.
- **별도 control unit 부재 — 단순 FSM 동작.**
  - PIM 의 핵심 이점인 명령어 디스패치 / 스케줄링 오버헤드 제거를 substrate 수준에서 직접 반영.
  - Tile 당 cycle 수 고정 → 실행 시간 결정론적 → 스케줄러가 PIM 완료 시점을 정확히 예측, GPU 연산과의 overlap 사전 계획 가능.
- **Internal path BW 우위 (vs GPU 외부 경로).**
  - **PIM 경로 (내부)** — row buffer → logic die SFU 내부만 사용, 외부 버스 오버헤드 없음 → channel peak × 100% 활용.
  - **GPU 경로 (외부)** — row buffer → TSV → interposer → GPU 메모리 컨트롤러 → SM. 직렬화 지연 · 컨트롤러 큐잉 · 인터포저 레이턴시로 peak 대비 손실 (η_HBM < 1, Phase 0 Discovery 산출).
  - **결과** — SP-PIM 2048 채널 aggregate 실효 BW 가 GPU aggregate 실효 BW 를 (1 / η_HBM) 배 초과. Substrate-level 자유도가 닫혀 거의 확정값 (정량 = Phase 3 종료 후 공개).

### 3.2 Channel-level PIM Toggle

HBM4 1 스택당 32 채널 각각에 대해 PIM / 일반 모드를 독립적으로 토글할 수 있다.

- **Scale** — GPU 당 8 스택 × 32 채널 = **256 채널** (per-GPU), Instance A 8 GPU 합계 **2048 채널** (aggregate) 이 독립 토글 가능.
- **Runtime split** — 임의 시점에 k_total 채널은 PIM 연산, 나머지는 GPU 측 트랜잭션 수행.
- **운영 단위 = 스택** — per-GPU k = n × 32, n ∈ {0, 1, …, 8}.
- **SP-PIM 극단 케이스** — Instance A 전체에서 k_total = 2048 을 활성화하여 단일 attention 연산을 8 GPU 간 lock-step 협력 처리 (§3.4 참조).

### 3.3 KV Cache Placement

KV cache 는 Instance A 의 8 GPU 에 분산 저장.

**GPU 간 sharding — architectural 강제 아님.**

- **옵션** — Head-sharded (GQA 8 KV head ÷ 8 GPU, head 1개/GPU), sequence-sharded (KV row 분산 + partial output reduce), hybrid 모두 동일 attention 결과 도달.
- **기본 채택** — Head-sharded (통신 비용 최소, output reduce 불필요).
- **확장 호환** — TP > KV head 수 환경 (예: TP=16 with GQA=8) 으로 확장 시 sequence-sharded / hybrid 로 자연 전환 가능.

**SP-PIM 채널-수준 KV-row sharding — GPU 간 sharding 결정과 직교.**

- **채널 자기-완결** — 각 GPU 256 채널에 32-way row-striped, aggregate 2048 채널. 어떤 GPU 간 mapping 위에서도 채널이 자기 KV row slice 를 self-contained 하게 sweep.
- **불변성** — Layer 단위로 동일 매핑 반복, KV 는 Instance A 의 HBM 에 영구 보존 (인스턴스 간 KV 전송 없음).

### 3.4 Instance Disaggregation: Attention Block vs FFN Block

트랜스포머 레이어를 attention block 과 post-attention block (FFN) 으로 분리하여 별도 인스턴스에 배치 (per-layer 14-step flow 및 instance 매핑은 README 의 [`instance_disaggregation.png`](figures/instance_disaggregation.png) 참조).

- **인스턴스 간 연결** — inter-instance pipeline 으로 직렬.
- **인스턴스 내부 병렬** — Tensor Parallelism (TP).
- **Instance A 추가** — PIM channel sequence parallelism (SP-PIM).

**채택 구성 (Case A) — 총 16 GPU**

| Instance | GPU | 내부 병렬 | 담당 연산 |
|---|---|---|---|
| A | 8 | TP=8 + SP-PIM (2048 channel Q-replicate) | input_layernorm, QKV projection, RoPE, KV save, attention (PIM), O projection |
| B | 8 | TP=8 | post_attention_layernorm, FFN gate/up/down, residual add |

**Instance A — Attention Block (GPU + PIM)**

- GPU 구성: 8 GPU, TP=8. GPU 간 KV sharding 정책은 §3.3 참조.
- 전체 layer 를 8 GPU 가 함께 처리 (각 GPU 는 layer 마다 1/8 weight 보유).
- PIM SFU: 각 GPU 의 HBM logic die 에 탑재. GPU 당 8 스택 × 32 채널 = 256 PIM 채널. 8 GPU 합계 **2048 채널**.
- SP-PIM: 단일 attention 연산을 8 GPU 의 2048 채널 전체에서 lock-step 협력 처리. Q 를 broadcast 하고 KV-row 를 채널에 sharding 하여 BW 를 aggregate. GPU 간 KV sharding 방식과는 직교 (§3.3).
- KV cache: Instance A 의 HBM 에 영구 보존. 인스턴스 간 전송 없음.

**Instance B — Post-Attention Block (non-PIM)**

- GPU 구성: 8 GPU, TP=8. Instance A 와 동일한 TP 폭으로 통신 simplicity 확보.
- 담당: residual add + post_attention LN + FFN (gate, up, down) + residual add.
- Instance B GPU 수: **8 채택**. Long-ctx 영역에서 A_cycle (PIM attention + projection overlap) 과 B_cycle (FFN TP=n) 이 A-bound 로 자연 전환되므로 n 추가 확대의 per-GPU SLO goodput 개선이 제한적.

**Instance B Memory Substrate — Hardware Cost Reduction**

Instance B 의 메모리 요구사항은 Instance A 와 구조적으로 다르다.

- **접근 패턴.** Instance A: KV cache 가 요청 수 × 문맥 길이에 비례하여 HBM 에 누적, 대역폭과 용량 모두 SOTA HBM 이 요구됨. Instance B: layer 별 FFN 가중치 (FP16, gate + up + down GEMM weight) 만 보유. KV 축적 없음, 크기 고정.
- **Compute-bound 판정.** FFN 은 compute-bound. HBM 대역폭이 FFN 처리 시간의 병목이 아니므로 HBM4 고대역폭 (≥ 4 TB/s) 이 B_cycle 에 기여하지 않는다.
- **저비용 substrate 대체 가능성.** FFN 가중치 streaming 시 compute-bound 특성 유지하는 영역에서, 두 옵션이 가능:
  - **(a) GDDR (GDDR6 / GDDR6X) 대체** — Substrate technology 자체 변경. 용량 요건 (TP=8 기준) 도 GDDR 표준 모듈 (24 GB) 로 충족. HBM4 8 stack 대비 unit cost 대폭 절감 (per-GB 3-5×) + packaging cost (interposer + CoWoS) 추가 절감. Trade-off: per-bit 전력 2-3× 상승 (긴 PCB 경로 + 높은 clock + termination loss), 단 compute-bound regime 의 낮은 BW utilization 으로 부분 mitigated.
  - **(b) 적은 stack 수 HBM** — 동일 substrate technology, stack 개수만 reduce (예: 8 stack → 2-4 stack). HBM 의 power 효율 (3-5 pJ/bit) 보존. Trade-off: 절감 폭 한정 — packaging cost 일부 보존.
  - 양 옵션 모두 B_cycle 에 영향 없음 (compute-bound regime 유지). 선택은 cost / power / supply availability 의 trade-off 영역.
- **비교 공정성 유지.** Instance B memory substrate 변경 (양 옵션) 은 B_cycle 에 영향을 주지 않으므로 (compute-bound 유지) PULS vs baseline 비교 ratio 보존. 정량 분석 (필요 module 수, cost / power 비율, sweet spot stack 수) 은 Phase 3 영역.

**TP+SP 선택 근거 (PP 기각)**

**SP-PIM × PP 양립 불가.** SP-PIM 은 단일 attention 연산을 시점 t 에 2048 채널 전체로 분산 처리한다. 이는 모든 GPU 가 동일 layer attention 에 lock-step 참여해야 성립한다. PP 는 정의상 시점 t 에 GPU 마다 다른 layer 처리이므로 SP-PIM 과 architectural 충돌. PIM 의 BW 우위를 leverage 하려면 TP 가 강제된다.

**인스턴스 간 데이터 트랜잭션**

A 와 B 는 inter-instance pipeline 으로 직렬 연결된다. 매 layer 마다 A → B → next layer A 순서.

| 방향 | 데이터 | 비고 |
|---|---|---|
| A → B | O projection output `[B × hidden]` | NVLink 4 SXM, intra-node send_recv |
| B → A | FFN output `[B × hidden]` (next layer input) | 동일 substrate |
| KV cache | **전송 없음. Instance A HBM 고정** | — |

레이어당 2 회, L-layer 모델 기준 forward pass 당 2L 회. 비동기 transfer 로 A/B 연산 시간 내 hiding 가능.

**Pipeline 구조**

```
[μ-batch M]    A(layer i) ─ NVLink ─ B(layer i) ─ NVLink ─ A(layer i+1) ─ … ─ B(layer L)
[μ-batch M+1]  A(layer i-1) ─ … (steady-state overlap)
```

- A 와 B 는 서로 다른 micro-batch 를 동시 처리 (instance-level pipeline).
- Steady-state cycle = max(A_cycle, B_cycle).
- L layer 통과 = L × cycle.

상용 deterministic compute scheduling 사례와 동일 design rationale (PIM FSM 의 결정론적 timing 이 pipeline scheduling 의 예측 가능성을 확보).

### 3.5 Host↔PIM Interface (Interceptor)

**Interceptor 의 본질 — attention 한정 scope 의 직접적 귀결.**

- **Mechanism — PHY DQ 경로 위 MUX** — Interceptor 는 평소 GPU 로 향하던 DQ 출력의 *행선지만* PIM SRAM 으로 가로챈다. Bank read sequence · timing 은 표준 JEDEC RD / WR 그대로 — 본질적으로 GPU 로 데이터 빼내는 동작과 동일하며, 단지 PHY 단에서 방향만 분기.
- **인터페이스 비용 0** — Decode-attention 단일 연산으로 scope 가 한정 (P2) 되어 JEDEC RD / WR 명령의 RFU bit 1 개만 점유. 추가 명령 · 디코더 · 큐 · 스케줄러 모두 불필요.
- **기존 HBM-PIM 과의 대비** — PIM scope 가 일반 GEMV 까지 확장된 기존 아키텍처는 이 단순성에 구조적으로 도달 못 함.

#### 3.5.1 Start (Interceptor)

GPU 가 기존 DRAM 명령 (RD/WR) 의 **RFU (Reserved For Future use) bit 1 개를 PIM_toggle 로 영구 점유**. PIM_toggle = 1 인 명령은 logic die 의 **Interceptor (DQ 출력 경로 위 MUX)** 가 데이터 행선지를 PIM SRAM 으로 가로챈다 — bank read 자체는 표준 JEDEC RD/WR sequence 그대로.

- **Layer 시작 metadata** — `num_tiles` + `mode` (FP16/FP8) 만 RFU 비트로 추가 전달
- **주소 필드 재사용** — KV / Q / output 주소는 GPU 가 어차피 발행하는 RD/WR 명령의 주소 필드로 자연 전달
- **별도 명령 추가 = 0 개** — JEDEC HBM4 표준 명령 집합 외 신규 opcode 정의 불요

#### 3.5.2 End (Computed Wait)

- **FSM determinism** — 1.3 GHz clock 위 tile 당 고정 cycle, jitter ±0 (§3.1 참조).
- **Computed wait** — GPU 가 PIM_toggle 발사한 순간 종료 시각을 사전 계산 → 정시에 HBM 에서 결과 read.
- **별도 동기 메커니즘 불필요** — Completion notification · interrupt · barrier 모두 불요.

§5.3 overlap policy 와 §6.3 scheduler dispatch 의 *"PIM 완료 시점 사전 계산"* 가정의 직접적 구현 기반.

#### 3.5.3 결과 전달 — HBM 경유

- **유일한 채널 = HBM** — PIM 은 HBM4 logic die, GPU 는 별도 die 에 위치하여 직접 P2P 통신선 없음.
- **Write → Read 프로토콜** — PIM 이 결과 O 를 HBM 의 정해진 주소에 **write** → GPU 가 computed wait 로 정시에 그 주소를 **read**.
- **GPU 측 정합 자연 산출** — GPU 내부 kernel 간 데이터 전달도 동일 방식 (global memory 경유) → 별도 DMA 엔진 · doorbell 메커니즘 불요.
- **PIM-GPU TSV 대역폭 contention 마진** — PIM (HBM4 logic die 내부) 과 Instance A GPU 가 HBM TSV 대역폭 공유로, 동시 full-load 동작 시 상호 throttling 가능. PIM decode-attn 예측 시간 위 GPU prefill chunk 산출 시 10% conservative 시간 마진 (`PIM_SLACK_SAFETY_MARGIN = 0.9`) 적용 → contention 방지. Stage 2 / Impl-11 위 calibrated 값 refinement.

## 4. Op Partitioning

| Operation | Phase | Executor | Rationale |
|---|---|---|---|
| Attention | **Decode** | **SP-PIM (Instance A)** | Memory-bound, KV streaming, 가변 길이 흡수 — P2 정합 |
| Attention | **Prefill** | **GPU attention kernel (Instance A)** | Compute-bound, tensor core 우위, PIM compute density 부족으로 부적합 |
| QKV / Output projection | All | GPU (Instance A) | Compute-bound, tensor core 우위; logic die 면적 제한으로 PIM 탑재 비현실적 |
| FFN | All | GPU (Instance B) | Compute-bound, 가중치 규모로 logic die 면적 비현실적 |
| LayerNorm / Softmax / RoPE / Activation | All | GPU | Negligible compute, kernel launch overhead |

**왜 decode attention 인가 — Positive Fit Rationale.** PIM substrate 위에서 가속이 성립하려면 다음 3 조건이 동시 충족되어야 한다:

1. **Online streaming 가능** — KV 를 한번에 전부 로드하지 않고 row-wise 로 sweep 하며 처리. Tile 단위 FSM 으로 메모리 path 점유 시간을 결정론적으로 분할 가능 (§3.1 의 32-row tile FSM 구조와 정합).
2. **Reduction 구조 — O(1) 내부 상태** — 중간 결과가 logic die SFU 내부에 O(1) (head_dim 크기) 로 누적되어 logic die ↔ DRAM die 왕복 트래픽 회피. Softmax denominator + row max 누적이 정확히 이 구조 (FlashAttention 의 online softmax 알고리즘).
3. **대량 MAC 적재 불필요** — Reduction tree / dense matrix 연산이 아니라 head_dim × tile_size 크기의 좁은 MAC sweep 으로 충분. Logic die 의 면적 제약 (P1 의 KV cache 비침범 전제) 과 양립.

Decode attention 만 이 3 조건을 동시 충족 → PIM scope 가 substrate 의 *positive fit* 에서 도출.

## 5. Scheduler Integration

### 5.1 Phase-aware Channel Activation

Instance A 의 SP-PIM aggregate 채널 수는 k_total = 2048 으로 고정. PIM 은 decode-attn 일이 존재하는 한 *항상* 가동되어, Instance A GPU 의 compute-bound 영역 (QKV · prefill_attn · O-proj) 의 HBM idle 헤드룸 위에 자연 overlap (O3 + §3.5.3). Sequence-parallel 성질 위 임의 시점 한 mb 의 decode-attn 이 모든 채널 점유 — 채널 분할 micromanagement 불필요 (Hermite identity 위 partition·serialize 동치). 잔여 TSV contention 은 10% margin `PIM_SLACK_SAFETY_MARGIN = 0.9` 으로 보수 흡수 — channel knob 부재 (Impl-10-pre-2).

- **Attention step** — Mixed batch 의 prefill chunk 토큰은 GPU attention kernel 이, decode 토큰은 SP-PIM 이 *동시 처리*. decode 토큰 존재 시 2048 채널 lock-step 단일 op. Pure prefill 배치 (decode rows 0) 는 PIM op_time = 0.
- **Projection step (QKV / O-proj / FFN)** — 같은 mb 의 PIM 작업 없음. Intra-instance double-buffering (§5.6) 위 *다음 mb* 의 decode-attn 이 projection 구간에 자연 overlap — P5 compute-bound timing 활성화 원칙 정합.

### 5.2 Fixed-shape Handoff to Instance B

PIM 이 attention 단계에서 KV 길이 의존성을 흡수하므로, Instance A → Instance B inter-instance 핸드오프 텐서 (§3.4) 가 항상 고정 형상이 된다.

- Decode batch: B × hidden
- Uniform-chunk prefill batch: (B · chunk) × hidden

Instance B 의 GPU 는 ragged batching 처리 없이 균일한 FFN GEMM 만 수행하므로 배치 내 straggler bubble 이 제거된다. (Instance A 의 GPU 는 prefill chunk attention 의 길이 의존성을 여전히 다루므로 본 효과의 직접 수혜 영역 아님.)

### 5.3 Compute-bound 구간 중 PIM 연산 Overlap

Instance A 내에서 GPU projection 이 compute-bound 상태일 때 HBM 대역폭이 부분적으로 idle 해진다. SP-PIM 은 이 헤드룸을 활용하여 다른 micro-batch 의 decode attention 을 overlap 처리한다 (intra-instance double-buffering, §5.6).

- **관찰 (전제):** QKV/O projection 및 FFN compute-bound 구간에서 HBM 대역폭 활용률이 낮아진다 (O3 참조). 이 idle 헤드룸이 SP-PIM 의 overlap 가능 영역.
- **메커니즘 (수단):** GPU op 진입 시점에 PIM 채널을 활성화하고, FSM 완료 시점 (결정론적 cycle 수) 을 사전 계산하여 GPU handoff 타이밍을 조율한다. 채널 단위 독립 토글 (§3.2) 로 GPU 명령 스트림과 충돌하지 않는다. 상세 동작 = §5.6.
- **Inter-instance pipeline 정합 (효과):** Instance A–B 간 pipeline cycle (max(A_cycle, B_cycle)) 이 PIM FSM 의 결정론적 타이밍으로 예측 가능해지므로, 마이크로 배치 스케줄링에서 SP-PIM overlap 창과 A↔B 데이터 전송 timing 을 정밀하게 배치할 수 있다.

**SP-PIM 분산 메커니즘.**

- **Q-replicate / KV-row sharding** — Q 를 k_total 채널 전체에 broadcast, KV row 를 채널에 sharding → 각 채널이 자기 KV slice 를 독립 sweep (§3.4 참조).
- **시간 산출** — Prefill chunk / decode batch 양 시나리오에서 channel 당 tile 수가 결정 → tile 수 × tile 시간 = SP-PIM attention 시간.
- **GPU baseline 대비 ratio** — §3.1 internal path BW 우위 (1 / η_HBM 배 초과) 와 ctx 종속 KV variance 의 결합으로 결정. **정량 산출은 Phase 3 sim 종료 후 공개.**

구체적 스케줄링 정책의 정량 평가는 Open Empirical Work (§8 E6) 참조.

### 5.4 스케줄링 예측 가능성의 부분적 해소

- **KV 길이 분산 흡수:** Decode 배치 내 요청별 KV 캐시 길이의 분산은 attention 연산 시간의 분산을 유발하여 straggler bubble 을 발생시킨다 (O1 참조). PIM 이 가변 길이 attention 을 흡수하면 Instance B 에는 항상 고정 형상 텐서가 전달되므로 (§5.2 참조), 이 불규칙성이 제거된다.
- **Prefill 우선 스케줄링 stall 제거:** Mixed batch 환경에서 prefill 연산이 우선 처리될 경우 decode 요청의 지연이 불규칙하게 발생한다. PIM 이 attention 의 길이 의존성을 흡수하면 prefill 이 decode 를 stall 시키는 원인이 제거되어, 동일 배치 내에서 prefill 과 decode 의 공존이 가능해지고 decode 지연의 불규칙성이 완화된다.

### 5.5 Prototype Vehicle: Self-authored Scheduler Framework

Scheduler core 는 self-authored event-driven framework 로 구현. OSS 코드베이스 (vLLM · Sarathi-Serve) 는 baseline scheduler reimplementation 의 reference 로만 활용하며 코드 의존은 없다.

- **Framework 구조:** Event queue + dependency DAG + in-flight μ-batch window 의 self-contained 자료구조. Production scheduler step 과 동일 호출 주기. Attention 호출은 PIM executor 로 라우팅하고, layer 를 Instance A (attention + projection) ↔ Instance B (FFN) 두 인스턴스로 분리 dispatch (§3.4).
- **Channel control:** Phase 진입 시 PIM 채널 수 *k* 를 scheduler step 에서 토글한다. Chunked prefill 정책과 직교적으로 호환.
- **TP=8 + SP-PIM 통합:** Instance A 의 GQA 8 KV head × TP=8 mapping 위에 SP-PIM Q-replicate 추가. Attention kernel 은 SP-PIM substitution 으로 구현.

### 5.6 Intra-instance Double-Buffering

Instance A TP=8 cluster 내에서 GPU projection 과 SP-PIM attention 을 μ-batch 단위 staggering 으로 동시 점유.

- **동일 μ-batch 내 — 순차 강제** — QKV projection (GPU) → attention (PIM SP) 은 Q 의존성으로 순차 실행. Inline overlap 불가.
- **상이한 μ-batch 간 — 의존성 부재** — μ-batch M 의 attention 을 SP-PIM 이 처리하는 동안 μ-batch M+1 의 QKV projection 을 GPU 가 선처리 가능. 이 비대칭이 double-buffering 의 architectural 근거.

**성립 조건.**

- **채널 분리 점유** — KV cache 용 채널 (PIM 모드, SP-PIM attention) 과 weight streaming 용 채널 (GPU 모드, projection) 이 HBM4 2048 채널 안에서 비중첩 부분집합 점유.
- **HBM 버스 경합 회피** — Projection 은 weight 채널, SP-PIM attention 은 KV 채널을 독립 사용 → 채널 분리 성립하는 한 완전한 overlap.

**기대 효과.**

- **Cycle 단축** — Instance A μ-batch 처리 시간 t_proj + t_attn → max(t_proj, t_attn).
- **Short-ctx** — projection bound, PIM attention hiding.
- **Long-ctx** — PIM attention bound, projection hiding.

**구현 요건.** Instance A 가 동시에 두 micro-batch 의 activation buffer 를 보유해야 하므로 activation footprint 가 2× 증가한다. 이 메모리 trade-off 는 추후 정량화 (Open Work).

### 5.7 가속 Source 분해

PULS 의 전체 가속 기여는 op-level 과 systems-level 의 다섯 소스로 분리된다.

| ID | 소스 | 메커니즘 | 주된 영역 |
|---|---|---|---|
| F1 | SP-PIM attention | 2048 channel Q-replicate, 단일 attention 을 8 GPU lock-step 협력. GPU attention kernel → SP-PIM, long-ctx 에서 ratio 보존 | Layer 처리 시간 감소, long-ctx 에서 압도적 |
| F2 | Projection ‖ PIM attention double-buffering | 서로 다른 micro-batch 의 QKV/O projection (GPU) 과 SP-PIM attention 동시 실행, A_cycle = max(t_proj, t_attn) | Instance A internal cycle 단축 |
| F3 | Instance A–B inter-instance pipeline (PB1 제거) | Steady-state 에서 A (attention + proj) 와 B (FFN) 가 서로 다른 micro-batch 동시 실행. 단일 instance 에선 attention → FFN 순차 처리 강제 | Layer 당 유효 시간 t_A + t_B → max(t_A, t_B) |
| F4 | μ-batch staggering | F2·F3 의 steady-state 전제 (별도 기여 항목 아님) | 전 영역 |
| F5 | PB3 제거 (channel-independent PIM scheduling) | SP-PIM 채널이 KV 길이에 독립적으로 동작하므로 batch 내 max-KV straggler bubble 무효화. Trace-grounded 분산 (axis 별 분해): (i) 공개 long-ctx agentic production trace + mid-ctx production chat trace, (ii) 1M-class real-doc benchmark dataset (long-ctx production trace 부재 영역의 대안). | SLO goodput, long-ctx production |
| (보조) | Mixed batching 복원 | Prefill + decode 동일 배치 내 가중치 공유로 arithmetic intensity 상승 | TTFT / throughput trade-off |
| (보조) | Bus traffic 절감 | PIM 이 attention 처리 시 HBM-GPU bus transaction 감소. Long-ctx 에서 큰 절감 | Energy / cost |

## 6. Instance A Scheduler 내부 정책

Instance A 내부 GPU·PIM 공동 스케줄링 정책.

- **Dispatch 위치** — chunked-prefill + 혼합 배치 mixed batch primitive 위에 PULS attention dispatch 를 얹는 형태 (§5.5).
- **Weight 공유 / attention split** — 동일 μ-batch 내 prefill·decode 요청이 QKV/O proj/FFN weight 를 공유하되 attention 만 token-type (prefill chunk → GPU kernel / decode → SP-PIM) 으로 split (§5.6).

**핵심.** 고정 슬롯 스케줄을 거부하고 *event-driven dispatch + 2-μ-batch lookahead* 를 채택한다. Adaptive admission 이 PULS 의 systems-level 차별성이며, 슬롯 경계가 이 자유도를 잠식해선 안 된다.

### 6.1 μ-batch 구성

μ-batch 는 서로 다른 요청을 phase 혼합으로 포함한다. KV cache 는 요청별 분리, weight (QKV proj, O proj, FFN) 는 모든 토큰이 공유. Per-layer QKV proj 와 O proj 는 μ-batch 전체에 일괄 GEMM, attention 만 token-type 분기 (prefill chunk → GPU kernel / decode → SP-PIM).

### 6.2 Invariants

스케줄러가 어떤 정책을 채택하든 다음 5 개 규칙은 위반 불가. I1·I2 는 *correctness invariant* (데이터 의존), I3 는 *efficiency invariant* (분할 O-proj 도 수학적으로는 동치이나 weight reuse · MFU · kernel launch 손해로 절대 분할 금지), I4·I5 는 *하드웨어 자원 제약*.

| ID | 종류 | 규칙 |
|---|---|---|
| I1 | correctness | prefill-attn(X) → QKV(X) 완료 후에만 dispatch (Q, K, V 의존) |
| I2 | correctness | decode-attn(X) → QKV(X) 완료 후에만 dispatch (PIM, FSM 시작 조건) |
| I3 | **efficiency** | O-proj(X) → prefill-attn(X) ∧ decode-attn(X) 모두 완료 후 dispatch. Row-wise 독립이라 분할 가능하나 `W_O` 2× streaming + MFU 하락 + 2× kernel launch 손해로 항상 단일 GEMM. Production mixed batch 표준 패턴 정합 |
| I4 | resource | GPU resource 는 시점 t 에 GEMM / attention op 하나만 실행 (tensor core saturate, kernel concurrency 무효) |
| I5 | resource | PIM resource 는 시점 t 에 decode-attn **op 하나만** 실행 (SP-PIM 이 2048 channel 전체 점유). Head · 요청 단위 제약 아님 — 단일 decode-attn op 내부에서 multi-head · multi-request batching 은 자유 |

### 6.3 Dispatch Policy: Event-driven + Dependency DAG

**정의.** Directed Acyclic Graph (방향성 비순환 그래프). 노드는 작업 단위, 엣지 `A → B` 는 *"A 완료 후 B dispatch 가능"* 의 precedence 관계를 의미. 비순환성으로 deadlock 불가, topological order 보장. PULS 는 §6.2 invariants 를 DAG 엣지로 코드화해 dispatch 결정을 *자료구조 위의 ready-node 선택 문제* 로 환원한다.

**노드.** 각 μ-batch `X` 에 대해 4 개 작업 노드: `QKV(X)`, `prefill-attn(X)`, `decode-attn(X)`, `O-proj(X)`.

**엣지.** I1·I2·I3 가 그대로 precedence 엣지가 됨.

| 엣지 | 출처 invariant |
|---|---|
| `QKV(X) → prefill-attn(X)` | I1 |
| `QKV(X) → decode-attn(X)` | I2 |
| `prefill-attn(X) → O-proj(X)` | I3 |
| `decode-attn(X) → O-proj(X)` | I3 |

**μ-batch 한 개 그래프.**

```
              ┌─→ prefill-attn(M) ─┐
QKV(M) ──────┤                    ├──→ O-proj(M)
              └─→ decode-attn(M) ─┘
```

서로 다른 μ-batch 간 명시적 엣지 없음 — 자원 (GPU·PIM) 가용 시 임의 인터리브 가능. 이것이 look-ahead / back fill 의 그래프-이론적 근거.

**Scheduler 사용법.** 스케줄러는 in-flight μ-batch window `{M_{i-1}, M_i, M_{i+1}}` 의 DAG 를 유지하고, 매 kernel 종료 이벤트마다 두 큐 (GPU·PIM) 에서 ready 작업을 dispatch.

```
on event(kernel K of μ-batch X completes):
    update DAG: mark node K(X) as done
    refresh ready set: 모든 precedence 가 done 인 미실행 노드
    GPU_next = pick(ready GPU jobs,
                    priority: O-proj > prefill-attn > QKV,
                    tie-break: oldest μ-batch first)
    PIM_next = pick(ready PIM jobs,
                    tie-break: oldest μ-batch first)
    if GPU idle: dispatch GPU_next
    if PIM idle: dispatch PIM_next
```

**`pick` 정의.** Priority queue dequeue 함수. 인자: (candidate set, priority rule, tie-break rule). 동작: candidate 중 priority rule 의 최우선 등급에 속한 노드들을 추리고, 그 안에서 tie-break rule 로 단일 노드 선택. 빈 candidate 면 null 반환 (이 경우 해당 자원 idle).

**GPU priority 순서 근거.** O-proj 지연은 *현재 μ-batch 완료 지연* 이므로 critical path 에 직접 영향. QKV 는 *다음 μ-batch* enable 작업이라 PIM idle 위험이 있을 때만 우선.

**Sync 보장.** I3 가 DAG 엣지로 자동 강제되므로 sync 누락 위험 없음. PIM FSM determinism 으로 dispatch 시점에 종료 시각이 계산되어, 다음 GPU dispatch 를 PIM 완료 시점에 맞춰 pre-scheduling 가능.

**Look-ahead / back fill 의 자연 산출.** I2 가 "QKV 완료 후 *언제든*" 으로 느슨하므로, PIM 이 idle 인 시점에 *GPU 가 어떤 μ-batch 를 작업 중이든 무관하게* 다른 μ-batch 의 decode-attn 을 시작 가능. 별도 정책 명시 없이 dispatch greedy 의 자동 산출.

### 6.4 Adaptive Admission

Per-μ-batch 구성 결정을 *iteration 단위로* 스케줄러가 동적으로 조정. 이전 iteration 의 GPU/PIM idle fraction 을 측정해 다음 μ-batch admission 을 조절한다. Chunked prefill 정책 위에 chunk size · decode batch 동적 조정 형태로 hook (§5.5).

**Decision Rule.** Admission control 의 본질적 frame:

- **Layer 1 — μ-batch 구성** (chunked-prefill + mixed batching primitive 위): prefill chunk vs decode 토큰 mix 및 N 결정. **TTFT / TBT SLO 의 결정 요인은 이 layer 에 응축**.
- **Layer 2 — DAG dispatch** (§6.3): Layer 1 결과 위에서 ready-node 자동 선택. 순차 처리이므로 admission 변수에 cancel — adaptive 자유도는 Layer 1 에 집중.

Adaptive admission 의 1차 objective = inter-instance pipeline cycle `max(A_cycle, B_cycle)` 의 두 instance 균형 (둘 다 fully utilized). 2차 objective = Instance A 내부 GPU·PIM double-buffering (§5.6) 의 균형. Hysteresis deadband 로 GPU jitter · workload variance oscillation 억제 (Deadband Policy 절 참조).

| Layer | 측정 | 진단 | Admission 조정 |
|---|---|---|---|
| Inter-AB (1차) | `A_cycle > B_cycle` (B idle) | A-bound (long-ctx) | admission ↓ 효과 제한적 (`A_cycle` 의 PIM attention 부분이 KV 길이 의존) — B idle 자연 수용 |
| Inter-AB (1차) | `A_cycle < B_cycle` (A idle) | B-bound (short-ctx + low batch) | prefill chunk admit → `A_cycle` 증가, 균형 회복 |
| Intra-A (2차) | GPU idle > `θ_high`, PIM busy | Instance A 내부 PIM 우세 | prefill chunk admit → GPU 윈도우 채움 (idle GPU 를 PREFILL_ATTN 으로 활용, PIM decode-attn 과 동시) |
| Intra-A (2차) | PIM idle > `θ_high`, GPU busy | Instance A 내부 GPU 우세 | decode 추가 admit → PIM 윈도우 채움 (idle PIM 을 decode-attn 으로 활용, GPU projection 과 동시) |
| — | 양 layer 모두 `θ_low` 이하 | balanced | 현재 admission 유지 |
| — | 양 layer 모두 idle | underloaded | μ-batch 크기 확대 또는 wait 토큰 가속 |

**Deadband Policy: Ctx-tiered Static Lookup.**

- **Width 산식** — Deadband width = `2σ_total` (control theory 표준, hysteresis 안정 조건).
- **`σ_total` 분해** — GPU jitter (L2 hit rate / warp scheduler / HBM controller queuing / kernel launch) 와 workload variance (KV 길이 분산 / arrival jitter) 의 RSS 합.
- **Ctx-tiered 채택 근거** — Long-ctx 일수록 cycle 길어 `σ` 누적이 커지고 KV variance 영향이 압도적 → ctx 별 정적 lookup.

| ctx | σ_total 추정 (정성) | deadband width |
|---|---|---|
| Short-ctx (2k–8k) | 낮음 (GPU jitter 지배) | 좁음 |
| Mid-ctx (~32k) | 중간 | 중간 |
| Long-ctx (128k–1M) | 높음 (KV variance 지배) | 넓음 (clamp 0% 영역 진입) |

`σ_total` 정량화 및 deadband sweep, online adaptive variant (per-iteration `σ` estimator 로 width 자동 갱신) 모두 본 연구 범위 밖 future work — self-authored scheduler framework 에 실 hardware jitter 모델 부재로 σ 측정 자체가 정의되지 않음. 본 평가는 GPU·PIM cycle 이 균형 영역 (balanced regime) 에서 dispatch policy 의 정성적 거동만 측정.

**Admission Lower Bound: MFU Floor.** `N ≥ N_sat` (FFN GEMM saturating knee) — 이 이하론 GEMM MFU sub-saturating, kernel launch overhead 지배. Upper bound 는 TPOT SLO model 영역 (future work). Ctx 별 binding:

| ctx 영역 | Binding |
|---|---|
| 2k–256k | B-bound (FFN GEMM saturating knee) |
| 256k–512k | Transition |
| ≥ 512k | A-bound (A_cycle 안에 B latency hidden) |

### 6.5 Example Dispatch Trace

PULS 스케줄러의 balanced steady state 에서 3 μ-batch in-flight window 의 한 instance (§6.4 에 의해 cycle balance 유지). 예시 구성:

| μ-batch | 구성 | 비고 |
|---|---|---|
| P | {X: prefill chunk (✓ Init 이전 완료), G, H: decode} | M 의 직전 μ-batch. Init 시점에 P 의 GPU stage 들은 이미 모두 끝났고 PIM 상에 decode-attn(G,H) 만 잔존 — 원래 phase 구성과 무관한 tail state |
| M | {A: prefill chunk, B: decode, C: decode} | 현재 μ-batch |
| N | {D: prefill chunk, E: decode, F: decode} | 다음 μ-batch |

아래 표는 event-driven dispatch 의 *한 trace* 이며 고정 주기가 아니다. `T_i` 는 kernel-completion dispatch event 시각, `Init` 은 trace 시작 시점의 active 상태.

| event | GPU 작업 | PIM 작업 | DAG state |
|---|---|---|---|
| Init | QKV(A,B,C) of M [back-fill: PIM 이 P 에서 busy 인 동안의 GPU emergent 활동] | decode-attn(G,H) of P [진행 중] | O-proj(P) not ready (I3, PIM 진행 중); QKV(M) 만이 ready GPU 노드 → priority dequeue 가 dispatch |
| T1 | O-proj(P) [PIM(P) 완료 trigger] | decode-attn(B,C) of M | PIM(P) done → I3 만족 → O-proj(P) 발화; M QKV done → I2 만족 → PIM 이 M decode dispatch |
| T2 | prefill-attn(A) of M | (decode-attn(B,C) of M 계속) | GPU O-proj(P) done → priority pick: prefill-attn(M) (O-proj(M) 는 PIM busy 라 아직 not ready) |
| T3 | QKV(D,E,F) of N [back-fill 재발현] | (decode-attn(B,C) of M 계속) | GPU prefill(M) done → O-proj(M) still not ready → priority 가 QKV(N) 로 떨어짐 |
| T4 | O-proj(M) [PIM(M) 완료 trigger] | decode-attn(E,F) of N | PIM(M) done → I3 만족 → O-proj(M); N QKV done → I2 만족 → PIM 이 N decode dispatch |
| T5 | prefill-attn(D) of N | (decode-attn(E,F) of N 계속) | T2 와 동일 패턴 — M cycle 의 GPU 파이프라인이 N 으로 반복 |

**G,H O-proj 처리 — GPU back-fill 의 emergent 속성.** PIM 이 P 의 decode-attn(G,H) 처리 중일 때 I3 가 O-proj(P) 를 not-ready 로 묶기 때문에 GPU 는 idle-wait 하지 않는다. Priority dequeue (`O-proj > prefill > QKV`) 에 의해 GPU 는 ready 인 유일한 노드 QKV(M) 를 pick 해 back-fill 로 처리한다. 따라서 T1 의 PIM 완료 trigger 는 *이미 M QKV 를 완료한 GPU* 로 떨어지며, 해방된 GPU 자원이 그 trigger 위에서 즉시 O-proj(P) 를 dispatch 한다. 이 emergent GPU back-fill 이 곧 §6.3 priority dequeue 의 실현 — 별도의 lookahead 정책은 인코딩되지 않는다. T3 에도 같은 패턴 (PIM 이 M 에 머무는 동안 QKV(N) back-fill) 이 반복된다. 만약 GPU 의 QKV(M) 가 PIM(P) 보다 길었다면 T1 이 GPU QKV 완료 시점으로 이동했을 것이며, balanced admission 하에서 몇 iteration 내에 re-equilibrate 되고, 어느 순서든 DAG 가 자동 처리한다.

**Regime applicability.** 위 trace shape 는 balanced admission 하의 PULS 스케줄러 steady-state attractor 이며 따라서 **ctx-independent** 이다. Chunked prefill (§5.5) 이 admission 으로 하여금 chunk granularity 를 조절해 `t_PIM(decode-attn) ≈ sum of GPU stages` 를 TBT SLO 안의 어떤 ctx 에서도 유지하게 하므로, chunk 크기와 무관하게 동일한 Init/T1–T5 dispatch 순서로 회귀한다. Trace 가 적용되지 않는 것은 cycle balance 자체가 불가능한 매우 긴 ctx (§6.6 "A-bound natural transition") 뿐이며 — 이는 ctx 가 길어질 때 나타나는 multi-request scheduler 의 시스템-레벨 본질적 한계이지 PULS 고유의 한계가 아니다.

### 6.6 Bound 분석

정성적 추정. 스케줄러는 §6.4 idle fraction 으로 bound 를 runtime 인식하므로 본 표는 control input 이 아니다. Sim 측정 후 component 시간 · transition ctx 만 갱신.

**Intra-A bound** — Instance A 내부 GPU stage (projection + AR) vs PIM stage (decode attention), double-buffering (§5.6) 관점.

- **Short-ctx (2k–8k): GPU-bound.** GPU stage 합 > t_PIM. §6.4 Intra-A 처방: decode 추가 admit → 다음 μ-batch PIM 윈도우 채움. AR cost 는 baseline 도 동일 부담이라 transition ctx 위치만 유지.
- **Long-ctx (≥ 32k): PIM-bound.** t_PIM > GPU stage 합. §6.4 Intra-A 처방: prefill chunk 추가 admit → 다음 μ-batch GPU stage 채움.

**Inter-AB bound** — Instance A pipeline cycle vs Instance B FFN cycle, inter-instance pipeline (§3.4) 관점. A_cycle = max(t_proj + t_AR, t_PIM) vs B_cycle = t_FFN_wide.

- **Mid-ctx (32k–256k): B-bound.** B_cycle > A_cycle. PIM attention 이 아직 FFN cycle 보다 짧음.
- **Long-ctx (≥ 256k): A-bound 자연 전환.** A_cycle ≥ B_cycle. PIM attention 이 FFN cycle 능가.

**Tile-level (per-tile, ns scale)** — PIM decode-attn 의 tile 시간 = max(FP8 load 시간, FSM compute 시간). FP8 KV 가정에서 compute-bound regime, FP16 KV 가정에서 load-bound regime 으로 이행하며 tile 시간이 약 2× 로 증가. FP8 quantize 가 본 t_decode-attn_PIM 의 직접 enabler.

### 6.7 구현 요건

- Self-authored event-driven framework: event queue 1 개, dependency DAG 1 개, in-flight window 3 μ-batch 상태. Production scheduler step 과 동일 호출 주기.
- PIM 종료 시각 predictor (FSM cycle-accurate).
- Idle fraction telemetry (GPU·PIM 별, iteration 단위 누적).
- Admission controller (chunk size · decode batch 동적 조정).

## 7. Orthogonality to Complementary Techniques

### 7.1 Paged KV Memory Management

- **계층 구분:** Paged KV 관리 기법은 KV 캐시를 비연속 메모리 페이지로 관리하는 *메모리 관리* 계층이다. PULS PIM 은 HBM 에 상주하는 KV 데이터 위에서 attention 연산을 실행하는 *컴퓨팅 오프로드* 계층이다.
- **비간섭:** 양자는 서로 다른 추상화 수준에서 동작하며 인터페이스를 공유하지 않는다. PIM FSM 은 페이지 테이블 참조를 통해 비연속 KV 레이아웃을 투명하게 처리할 수 있으므로, 페이지 기반 물리적 배치 결정은 PIM 연산의 정확성에 영향을 주지 않는다.
- **이득 누적:** 페이지 관리가 제거하는 단편화 손실과 PIM 이 제거하는 GPU-side attention 비용은 독립적으로 누적된다.

### 7.2 Speculative Attention

- **기법 정의:** Speculative decoding 은 draft 토큰 생성 후 단일 forward pass 에서 병렬 검증을 수행한다. Speculative attention 은 이 검증 패스의 attention 비용을 최적화한다.
- **PIM 적용 가능성:** PULS PIM 은 토큰 출처 (draft / verified / speculation tree 내 위치) 에 무관하게 attention 연산을 동일하게 처리하므로, speculative attention 패스에 대해서도 PIM offload 가 성립한다.
- **이득 합산:** speculative decoding 이 forward pass 횟수를 줄이고, PIM 이 각 pass 의 attention 비용을 낮춘다. 두 최적화의 결합 처리량 향상은 곱셈적으로 작동한다.

### 7.3 Prefix KV Caching

- **기법 정의:** Prefix 공유 KV 캐시 히트로 prefill 연산 자체를 생략하는 기법군.
- **히트 구간:** PIM 의 prefill-side attention 부하도 비례하여 감소한다.
- **미스 및 decode 구간:** PIM offload 이득이 그대로 유지된다.
- **이득 누적:** Prefix caching 이 KV 재사용으로 전체 연산 규모를 축소하고, PULS 가 남은 연산의 attention 비용을 흡수한다. KV 히트율과 PIM offload 이득은 독립 변수로서 교차 항 없이 성능에 기여한다.

## 8. Open Empirical Work

*시뮬레이션 공통 가정: 모든 요청은 KV 캐시 히트 없이 처음 입력되는 신규 요청으로 처리한다 (prefix 중복 없음). KV 히트가 발생하는 실제 워크로드에서는 prefill-side attention 부하가 추가로 감소하므로 PULS 의 상대적 향상 폭은 본 시뮬 결과보다 더 커질 것으로 예상한다.*

**E1.** SP-PIM aggregate (2048 channel) vs H100 GPU attention kernel 토큰 처리량 비교.

- **PIM token/s** — FSM 사양 (FP8 KV 전제, §3.1) 으로부터 도출.
- **GPU baseline** — H100 attention kernel (동일 FP8 KV 모드) 의 측정 token/s.
- **HBM4 추정** — H100 HBM3 측정치 × peak ratio scaling.

**E2.** LLM 서빙 phase 별 H100 HBM 대역폭 활용 곡선 측정 (cycle-accurate).

- **산출** — PIM 가용 대역폭 헤드룸 정량화.
- **HBM4 scaling** — H100 측정치 → HBM4 추정.

**E3.** Phase-aware 채널 분할 비율의 sensitivity sweep 과 최적 정책 도출.

**E4.** Mixed-batch chunk 크기 튜닝과 arithmetic intensity 향상의 정량 평가.

**E5.** Decode batch 의 straggler bubble 분석.

- **원인** — 요청별 KV 길이 분산이 attention 시간 분산을 유발 → batch 내 bubble.
- **측정** — Bubble 크기 정량화 (KV 길이 분산의 함수).
- **대조** — PIM 적용 전후 bubble fraction + 전체 처리량 감소분.

**E6.** Production-grade scheduler 의 compute-bound 구간 (FFN, prefill-side projection) headroom 측정.

- **점유 비율** — 전체 연산 시간 대비 compute-bound 구간 비중.
- **HBM 대역폭 활용도** — 해당 구간의 GPU HBM 활용률 동시 추출.
- **용도** — PIM overlap 가설 (§5.3) 의 가용 헤드룸 근거.

**E7.** PIM offload 에 의한 HBM-GPU 간 bus transaction 감소 정량 분석.

- **측정 항목** — GPU 측 HBM 트랜잭션 횟수 + 총 전송량 (PIM 적용 전후).
- **Phase 분해** — prefill / decode 별 트랜잭션 절감 분리 측정.
- **Inter-instance 비용 비교** — Instance A/B 간 `[B × hidden]` 전달 비용과의 대조.

---

본 architecture 문서의 정량 수치 (가속 배수, latency / throughput 절대값, MFU plateau, admission ceiling 수, deadband width %) 는 모두 **Phase 3 calibration 영역에서 측정 예정**.
