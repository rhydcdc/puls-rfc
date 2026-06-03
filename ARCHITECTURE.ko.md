# PULS Architecture

**P**IM-**U**nified **L**LM **S**erving — scheduler-aware co-design.

- Motivation / problem statement / proposal 개관 — [`README.md`](README.md) 참조
- 정량 source decomposition (Aux1·Aux2·F3·F5) — [`README.md`](README.md#results) 참조
- F1·F2 ablation + 절대 metric (TTFT / TPOT / throughput) — 후속 calibration 으로 연기 / silicon 부재로 out of scope

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
  - [6.4 Admission: 동작점 (풀 모델)](#64-admission-동작점-풀-모델)
  - [6.5 Example Dispatch Trace](#65-example-dispatch-trace)
  - [6.6 Bound 분석](#66-bound-분석)
  - [6.7 구현 요건](#67-구현-요건)
  - [6.8 2-active μ-batch 구성 검증](#68-2-active-μ-batch-구성-검증)
- [7. 클러스터 스케일: 노드 풀 100K 센터링 라우팅](#7-클러스터-스케일-노드-풀-100k-센터링-라우팅)
  - [7.1 동기: 센터링 없는 클러스터의 idle 폭발](#71-동기-센터링-없는-클러스터의-idle-폭발)
  - [7.2 노드 HBM 의 실제 구성](#72-노드-hbm-의-실제-구성)
  - [7.3 Cold-start: 엣지 게이팅 + interleave greedy](#73-cold-start-엣지-게이팅--interleave-greedy)
  - [7.4 Healing: 전략적 greedy refill (무축출)](#74-healing-전략적-greedy-refill-무축출)
  - [7.5 측정 결과 — E 스윕 · 안정성](#75-측정-결과--e-스윕--안정성)
- [8. Orthogonality to Complementary Techniques](#8-orthogonality-to-complementary-techniques)
  - [8.1 Paged KV Memory Management](#81-paged-kv-memory-management)
  - [8.2 Speculative Attention](#82-speculative-attention)
  - [8.3 Prefix KV Caching](#83-prefix-kv-caching)
- [9. Open Empirical Work](#9-open-empirical-work)

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
  - **GPU 경로 (외부)** — row buffer → TSV → interposer → GPU 메모리 컨트롤러 → SM. 직렬화 지연 · 컨트롤러 큐잉 · 인터포저 레이턴시로 peak 대비 손실 (η_HBM < 1, Discovery track 산출).
  - **결과** — SP-PIM 2048 채널 aggregate 실효 BW 가 GPU aggregate 실효 BW 를 (1 / η_HBM) 배 초과. Substrate-level 자유도가 닫혀 거의 확정값 (정량은 Aux2 / F3 산식에 진입 — [`README.md`](README.md#results) 참조).

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
- **비교 공정성 유지.** Instance B memory substrate 변경 (양 옵션) 은 B_cycle 에 영향을 주지 않으므로 (compute-bound 유지) PULS vs baseline 비교 ratio 보존. 정량 분석 (필요 module 수, cost / power 비율, sweet spot stack 수) 은 후속 calibration 으로 연기.

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
- **PIM-GPU TSV 대역폭 contention 마진** — PIM (HBM4 logic die 내부) 과 Instance A GPU 가 HBM TSV 대역폭 공유로, 동시 full-load 동작 시 상호 throttling 가능. PIM decode-attn 예측 시간 위 GPU prefill chunk 산출 시 10% conservative 시간 마진 (`PIM_SLACK_SAFETY_MARGIN = 0.9`) 적용 → contention 방지.

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

Instance A 의 SP-PIM aggregate 채널 수는 k_total = 2048 으로 고정. PIM 은 decode-attn 일이 존재하는 한 *항상* 가동되어, Instance A GPU 의 compute-bound 영역 (QKV · prefill_attn · O-proj) 의 HBM idle 헤드룸 위에 자연 overlap (O3 + §3.5.3). Sequence-parallel 성질 위 임의 시점 한 mb 의 decode-attn 이 모든 채널 점유 — 채널 분할 micromanagement 불필요 (Hermite identity 위 partition·serialize 동치). 잔여 TSV contention 은 10% margin `PIM_SLACK_SAFETY_MARGIN = 0.9` 으로 보수 흡수 — channel knob 부재.

- **Attention step** — Mixed batch 의 prefill chunk 토큰은 GPU attention kernel 이, decode 토큰은 SP-PIM 이 *동시 처리*. decode 토큰 존재 시 2048 채널 lock-step 단일 op. Pure prefill 배치 (decode rows 0) 는 PIM op_time = 0.
- **Projection step (QKV / O-proj / FFN)** — 같은 mb 의 PIM 작업 없음. Intra-instance double-buffering (§5.6) 위 *다음 mb* 의 decode-attn 이 projection 구간에 자연 overlap — P5 compute-bound timing 활성화 원칙 정합.

### 5.2 Fixed-shape Handoff to Instance B

PIM 이 attention 단계에서 KV 길이 의존성을 흡수하므로, Instance A → Instance B inter-instance 핸드오프 텐서 (§3.4) 가 항상 고정 형상이 된다.

- Decode batch: B × hidden
- Prefill batch: (prefill 총 토큰수) × hidden

Instance B 의 FFN GEMM 은 배치의 *총 토큰수* 만 의존하지 요청별로 토큰을 어떻게 쪼갰는지는 보지 않는다 — 따라서 요청별 prefill chunk 는 ragged 여도 무방하다(풀 모델 prefill steering 이 256 토큰을 불균등 분배해 depth-합을 맞춤, §6.4). 고정-형상 이득의 본질은 *attention* 단계의 KV 길이 분산을 PIM 이 이미 흡수해 Instance B 가 받는 형상이 토큰수에만 의존한다는 것 — 요청별 KV 길이 분산에서 오는 straggler bubble 이 제거된다. (Instance A 의 GPU 는 prefill chunk attention 의 길이 의존성을 여전히 다루므로 본 효과의 직접 수혜 영역 아님.)

### 5.3 Compute-bound 구간 중 PIM 연산 Overlap

Instance A 내에서 GPU projection 이 compute-bound 상태일 때 HBM 대역폭이 부분적으로 idle 해진다. SP-PIM 은 이 헤드룸을 활용하여 다른 micro-batch 의 decode attention 을 overlap 처리한다 (intra-instance double-buffering, §5.6).

- **관찰 (전제):** QKV/O projection 및 FFN compute-bound 구간에서 HBM 대역폭 활용률이 낮아진다 (O3 참조). 이 idle 헤드룸이 SP-PIM 의 overlap 가능 영역.
- **메커니즘 (수단):** GPU op 진입 시점에 PIM 채널을 활성화하고, FSM 완료 시점 (결정론적 cycle 수) 을 사전 계산하여 GPU handoff 타이밍을 조율한다. 채널 단위 독립 토글 (§3.2) 로 GPU 명령 스트림과 충돌하지 않는다. 상세 동작 = §5.6.
- **Inter-instance pipeline 정합 (효과):** Instance A–B 간 pipeline cycle (max(A_cycle, B_cycle)) 이 PIM FSM 의 결정론적 타이밍으로 예측 가능해지므로, 마이크로 배치 스케줄링에서 SP-PIM overlap 창과 A↔B 데이터 전송 timing 을 정밀하게 배치할 수 있다.

**SP-PIM 분산 메커니즘.**

- **Q-replicate / KV-row sharding** — Q 를 k_total 채널 전체에 broadcast, KV row 를 채널에 sharding → 각 채널이 자기 KV slice 를 독립 sweep (§3.4 참조).
- **시간 산출** — Prefill chunk / decode batch 양 시나리오에서 channel 당 tile 수가 결정 → tile 수 × tile 시간 = SP-PIM attention 시간.
- **GPU baseline 대비 ratio** — §3.1 internal path BW 우위 (1 / η_HBM 배 초과) 와 ctx 종속 KV variance 의 결합으로 결정. **정량 산출은 Aux2 / F3 산식에 진입 — [`README.md`](README.md#results) 참조.**

**PIM KV 대역폭이 정말 compute-bound 윈도우에 숨는가? (정량 근거).** *시간* 적합은 t_pim ≤ t_gpuA (OPERATING_POINT §2). *대역폭* 적합은 네 근거에 의존 — 셋은 spec 계산, 하나는 silicon-deferred:

1. **헤드룸은 실재.** GPU-A projection 은 compute-bound: QKV arithmetic intensity ≈ 349 FLOP/byte ≫ B200 roofline ridge (2200 TFLOPS ÷ 16 TB/s ≈ 137 FLOP/byte). 그래서 projection 중 외부 HBM 버스는 ~32% 만 사용 — ~68% 유휴.
2. **PIM 은 외부 버스를 아예 안 쓴다.** decode-attn 한 layer 가 읽는 KV = Σkv 6.15M × 2 KB (FP8 K+V, n_kv=8 · d_head=128; 배포 128) = 12.6 GB; t_pim ≈ 25 µs 안에 = **~500 TB/s — 외부 총대역 128 TB/s 의 ~4배**(비율은 prefill-invariant). 즉 KV 는 GPU 가 쓰는 버스로 *흘릴 수 없다* → PULS 는 DRAM row 에서 in-place 처리(F1 / Aux2 전제). "숨음"은 **남은 버스 대역에 끼어드는 게 아니라 경로 분리**.
3. **내부 경로는 더 빠르나 contention 0 은 아님.** attention SFU 가 로직다이에 있어 PIM 은 같은 채널 위 GPU 접근과 **TSV / 셀어레이 경로를 여전히 공유** — contention 존재. 단 외부 버스 프로토콜을 우회해 TSV 를 η_internal 로 받음, **≈ 1/η_external ≈ 1.35배** 의 유효 속도. 그리고 채널 토글(§3.2) 자체가 가속: normal-mode 채널이 GPU 에 가중치를 *주는 동시에* PIM-mode 채널이 decode-attn 을 먹임 — weight-load ‖ KV-compute 가 직렬이 아니라 동시 진행.
4. **silicon-deferred 폐쇄 (정직한 gap).** TSV / 셀어레이 대역폭이 GPU 잔여 weight-load 와 *동시에* PIM 전체 KV read 를 saturation 없이 버티는지는 η_internal · per-channel TSV 대역에 의존 — Ramulator2 *추정* 이지 silicon 측정 아님(OI9 / §3.5.3). 네 요인은 강한 근거이자 P5 + §3.2 아키텍처 전제이지 닫힌 측정은 아니다.

즉 PIM 은 **두 축**으로 숨는다: 시간(t_pim ≤ t_gpuA, OPERATING_POINT §2) + 대역폭(in-place 내부 경로 ~1.35× + 채널 동시성, 어차피 ~68% 버스-유휴인 GPU 윈도우로) — 정량 TSV-saturation 폐쇄는 silicon 대기.

구체적 스케줄링 정책의 정량 평가는 Open Empirical Work (§9 E6) 참조.

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

### 6.4 Admission: 동작점 (풀 모델)

스케줄러는 매 μ-batch 를 구성해 세 자원(PIM = decode-attn, GPU-A = projection + prefill-attn, FFN = Instance B)의 시간을 맞춰 inter-instance · intra-instance idle 을 최소화한다. 구성은 **세 관심사의 분리**이며, 각각 자기 in-flight 풀에서 독립적으로 steering 된다 — idle fraction feedback 으로 한 cohort 를 조절하는 게 *아니다.* (이전 초안은 측정 GPU/PIM idle fraction + hysteresis deadband 로 iteration 마다 admission 을 조정했으나 그 feedback 모델은 폐기 — 맺음말 참조. 이제 스케줄러는 측정 idle 에 반응하지 않고 *고정 타깃*에 steering 으로 명중한다.)

1. **Admission = 풀 보충만.** `request_queue → in-flight (PREFILL)`, aggregate KV budget(`can_admit`) 게이트만. decode/prefill 타깃은 보지 않는다. `prompt_len = 0`(decode-only)이거나 prefill 이 이미 끝난 요청은 즉시 DECODE 전이.
2. **decode-set steering.** in-flight **DECODE 풀**에서 두 타깃을 동시 명중 — **개수 62 ∧ Σkv 6.15M** (배포 128; 도출 기준 256 은 123·12.3M) — 하도록 로컬 그리디 steering + age-cap 으로 선택. 순수 *선택*(KV admit · 큐 조작 없음; KV 는 풀 진입 시 예약). 미선택은 age(`wait++`), 선택은 리셋(`wait = 0`).
3. **prefill steering.** in-flight **PREFILL 풀**에서 **128 토큰**(고정 FFN-batch knob, 아래 ①)을 멤버들에 분배해 PREFILL_ATTN depth-합이 **12.8M** 되게, 동일한 per-token 로컬 그리디 + age-cap. 0 토큰 받은 멤버는 *풀에 잔류*(빈 chunk 로 넣지 않음 — 넣으면 μ-batch 가 부풀어 decode 고갈) — decode 와 별개 축.

**이 세 관심사는 매 iteration 재실행된다 — per-iteration 재구성.** 매 forward pass 후 μ-batch 를 풀에서 재선택(`_recompose_mb`), 다른 활성 μ-batch 와 disjoint. in-flight window 는 활성 μ-batch 2 개 유지(`_STAGGERING_TARGET_MB = 2`; capacity 3 = 2 active + 1 전이 여유), 이는 F2/F3 overlap 의 staggering 전제(§5.6, §6.5). sticky-cohort 모델이 지운 자유도를 복원 — 구성이 풀의 drain/refill 을 따라간다.

> **이것은 실 continuous-batching 패러다임 그 자체이지 그 근사가 아니다.** 실서버는 생애 단계가 섞인 상주 인구를 유지한다 — 디코딩 중(긴 생성 진행), prefill 중(새 프롬프트), prefill 막 끝나 decode 로 전이한 요청 — 그리고 매 iteration batch 를 재구성한다(vLLM / Sarathi-Serve iteration-level scheduling). 풀 모델이 *바로 그 구조*다: PREFILL/DECODE 상태의 in-flight 풀, 풀 상주 KV, PREFILL→DECODE 전이, admission 보충, mixed batching 하의 per-iteration 재구성. 고정 cohort 를 얼려 함께 완료시킨 이전 sticky-cohort 모델이야말로 *비정합* 이었다 — 실서버는 batch 를 얼리지 않는다. 따라서 스케줄링 *메커니즘*은 구조적으로 정합이며, 합성으로 남는 건 *워크로드 공급*(warm-start seed + 풍부-풀 가정, README 에 정직 disclosure)뿐 — 메커니즘이 아니다. 그래서 §6.8 에서 검증하는 동작점 구성은 장난감이 아니라 *실 스케줄링 구조* 위에서 성립한다.

**동작점 (인과사슬 ① → ⑤).** prefill 토큰 수가 FFN batch 를 고정하고, 그것이 decode 개수를, 그것이 균형 ctx 와 함께 decode-KV 합을 고정한다:

| 순서 | 고정값 | 값 (prefill **128** 배포; 256 도출은 2배) | 조건 자원 |
|---|---|---|---|
| ① | prefill 토큰/배치 | **128** (2의 거듭제곱·커널 친화; 256 은 도출 기준) | GPU-A (PREFILL_ATTN = Σ chunk×depth) |
| ② | 균형 시간 X (= 산출주기) | **~25.5 µs** (X·L ≈ 2.0 ms = 배치 forward-pass 산출주기) | — |
| ③ | FFN batch | **190 토큰** | Instance B |
| ④ | **decode 개수 N_dec (제어 타깃)** | **62** (= 190 − 128) | Instance B |
| ⑤ | **decode-KV 합 (제어 타깃)** | **6.15M** | Instance A (PIM) |
| + | prefill KV-work (제어 타깃) | **12.8M** (= 128 × depth) | GPU-A |
| + | 균형 ctx | **~100K** (하드웨어 상수) | — |

*개수*(62)는 KV 길이와 무관(FFN 은 토큰 개수만), *KV 합*(6.15M)은 길이의 총합(PIM 은 합만). 둘 다 만족 ⟺ 평균 ctx ≈ 100K. (배포 128; 도출 기준 256 은 123·12.3M, 모두 2배 — OPERATING_POINT §1.)

**로컬 그리디 steering + age-cap (`former` 알고리즘).** 제어 타깃은 쌍 *(개수 62, Σkv 6.15M)* (배포 128; 도출 256 은 123·12.3M) — 단일 평균이 아니다. 순수 FIFO 는 Σkv 만 잡고 off-average 풀에서 개수를 놓친다(측정 spread 22–30%). 그래서 매 step *다음에 필요한 길이* 를 계산해 가장 가까운 디코더를 admit(steering); `≥ AGE_CAP` 기다린 요청은 강제 포함(age-cap — 공정성 / FIFO 의도). 전역 통계 · 미래 예측 없이 순수 로컬:

```
한 μ-batch (decode):                  # AGE_CAP = 5 (배포·클러스터; 옛 node-scheduler sweep 은 2)
  n=0, S=0
  while n < target_count(62) and S < target_kv(6.15M) and pool:
    if wait ≥ AGE_CAP 인 요청 있음: 가장 오래된 것 admit              # 공정성(강제)
    else: ideal=(target_kv−S)/(target_count−n) 에 가장 가까운 디코더 admit   # steering
  나머지 대기: wait += 1
  → (62, 6.15M) 수렴; n 단조 증가 → ≤62 step.
prefill: 128 토큰을 depth-합 12.8M 되게 동일 steering + age-cap 분배.
2 active μ-batch — 완료시 (반환분 + 잉여)로 재구성(capacity 3 = 2 active + 1 전이 여유).
```

- **★ 길이분산 무관 (핵심 성질).** 거대 변종 풀(실 트래픽)에서 짧은 거 + 긴 거를 *조합* 해 두 타깃 명중. 헤비 / 혼합 / bimodal — 어떤 분포든 동작, 평균을 안 보고 두 타깃만 맞추므로. age-cap 강제된 off-size 요청도 steering 이 보정(긴 거 강제 → `ideal`↓ → 다음에 짧은 거 다수)해 배치는 여전히 (62, 6.15M). 먼저 온 요청은 ≤ AGE_CAP+1 batch 안에 처리.
- **운영 파라미터 = target_count + target_kv + AGE_CAP.** ±10% 밴드 [5.55M, 6.77M] 는 **제어값이 아니라** 진단용 idle-SLA 라벨(밴드 폭 ≈ 허용 최악 idle: ±10% → edge idle ~8.6–10.6%). steering 이 타깃 명중하므로 실현 idle ≈ 0; `former` 는 밴드로 멈추지 않는다.
- **AGE_CAP 트레이드오프 (sweep).** cap↑ → steering 자유도↑ → spread↓, 단 대기(레이턴시)↑. cap↓ → FIFO化 → 공정 / 저지연이나 spread↑. `cap1: sp 3.1%` · `cap2: sp 1.2%, 대기≤3` · `**cap5: sp 0.7%, 대기5**` · `cap∞: sp 0.8% but starvation(대기37)`. → **AGE_CAP = 5 채택(배포)** — 대기 5 batch ≈ 128 µs ≪ TBT 라 레이턴시 무해, spread 0.7% < cap2. 옛 node-scheduler 는 보수적으로 2 였음(OPERATING_POINT §3).

**ctx 100K = 하드웨어 상수, 경험적 추측 아님.** 삼중 균형을 풀면 op-time 계수 비로 `ctx_balance = (K2+1)/K1`(PIM tile rate ÷ FFN flops/tok ÷ prefill-attn flops/tok·depth ÷ proj flops/tok); *prefill 이 약분돼 사라짐* → 균형 ctx 가 **모든** prefill 에서 100K(§5 스윕 B 실증). 역할은 타깃 *도출*(배포 128: Σkv 6.15M = 62 × 100K; 도출 256: 12.3M = 123 × 100K)이지 워크로드에 평균을 *강제* 하는 게 아니다. 이것이 길이분산 무관성의 근거 — 개별 요청 길이가 어떻게 분산되든 도출된 두 타깃만 맞춘다.

**prefill 배포 128 (도출 256, vs 512).** prefill 은 균형 ctx 가 아니라 *스케일 knob* X — 작을수록 산출주기·HBM 절반씩 줄고 TTFT / throughput 불변(X 가 prefill 에 선형이라 청크·cycle 상쇄), 단 FFN batch 가 MFU knee 위여야. 512→256→**128**: 산출주기 X 101→51→**25.5 µs**, HBM aggregate(decode) 60M→30M→**15M** = 9.8→4.92→**2.46 TB**(FP8 160KiB/tok). **128 만 64 공식 스택(4.40 TB)에 적합**(256·512 초과 — OPERATING_POINT §4.1). 유일 risk = FFN GEMM MFU 포화: wave-quant 추정상 batch ~128 포화, 128 배포의 batch = 62 + 128 = **190(> knee, 48% margin)** 이라 포화하나 256(379)보다 여유 적음; 모델 MFU = 0.6 고정이라 knee 미관측(silicon 부재). **배포 128, MFU 실측서 190 부족 시 256 복귀**(512 batch 759 더 안전, vLLM 수렴).

> **맺음말 (superseded).** 이전 초안의 idle-fraction-feedback + hysteresis-deadband admission 은 이 풀 모델로 대체됐다. deadband width 는 `2σ_total` 이었으나 σ 는 hardware jitter 모델 없는 self-authored framework 에서 측정 불가(§9 / OI4)라 feedback variant 는 애초에 작동 메커니즘이 아니었다. 풀 모델은 steering 으로 고정 타깃에 직접 명중; ±10% 밴드는 제어 입력이 아니라 진단용 idle-SLA 라벨로만 남는다.

### 6.5 Example Dispatch Trace

PULS 스케줄러의 balanced steady state 에서 **2 active μ-batch + 전이 tail**(capacity 3) in-flight window 의 한 instance (§6.4 에 의해 cycle balance 유지; 3개를 *구성*하는 게 아니라 2 active 가 overlap 하고 직전 batch 가 tail 로 빠지는 파이프라인). 예시 구성:

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

### 6.6 Bound 분석 — 제거 (동작점은 균형, single bound 없음)

옛 ctx-의존 bound 분석(short-ctx GPU-bound / long-ctx PIM-bound / mid-ctx B-bound)은 **steering 이전** 프레이밍이었다. §6.4 동작점은 composition 을 (62, 6.15M)·(128, 12.8M)에 steering 으로 명중시켜 세 자원 시간을 일치시킨다 → **단일 binding 자원이 없다**(idle ≈ 0 — §6.4 동작점 균형). 따라서 "어느 자원이 cycle 을 bound 하는가"는 동작점에서 무의미하므로 본 분석은 제거.

유일 예외 = **퇴화 극단 (A-bound 자연 전환).** 매우 긴 ctx-only 트래픽이라 짝지을 짧은 요청이 고갈되면 PIM attention 이 GPU-A 윈도우에 못 숨어 A_cycle ≥ B_cycle 로 넘어간다 — 실 무한-변종 트래픽엔 미발생, ctx 가 길어질 때의 multi-request 스케줄러 시스템-레벨 한계이지 PULS 고유 한계가 아니다(OPERATING_POINT §6). (FP8 KV 가 tile 시간을 compute-bound 로 유지해 t_decode-attn_PIM 을 절반으로 — substrate enabler 는 §3.2.)

### 6.7 구현 요건

- Self-authored event-driven framework: event queue 1 개, dependency DAG 1 개, in-flight window 2 active μ-batch (+ 전이 tail, capacity 3) 상태. Production scheduler step 과 동일 호출 주기.
- PIM 종료 시각 predictor (FSM cycle-accurate).
- Idle fraction telemetry (GPU-A · PIM · FFN 별, measurement window 단위 누적).
- 풀 모델 composer (decode-set steering ‖ prefill steering ‖ admission 보충, §6.4).

### 6.8 2-active μ-batch 구성 검증

**다배치 구성 — 2 active μ-batch (배포 모델).** window 는 2 active + 1 전이 여유(capacity 3). 한 batch 의 forward pass 가 끝나면 **(반환분 + 잉여)로 *재구성*만 하지, 3번째를 강제 구성하지 않는다.** 통합 lifecycle([cluster_lifecycle.cpp](implementation/analysis/cluster_lifecycle.cpp))이 prefill→decode 종속성·age-cap = 5 포함 배포 128 에서 **디코드(62 ∧ Σkv 6.15M)·프리필(128 ∧ depth-work 12.8M) 둘 다 100% 명중 · Σdev 0.20% / 0.07%**, **age-cap 꼬리 없음**을 검증(로직은 스케일 불변, prefill 256↔128 동형).

## 7. 클러스터 스케일: 노드 풀 100K 센터링 라우팅

§6 의 동작점(배포 128: 개수 62 ∧ Σkv 6.15M; 도출 256 은 123·12.3M)은 한 노드의 in-flight 풀이 **~100K 로 센터된 변종 풀**일 때 steering 이 명중한다(§6.4). 단일 노드는 admission 이 그 풀을 유지하지만, **서버스케일 클러스터**(노드 수백–수천)에선 글로벌 도착 평균이 100K 보다 높아 노드별 풀이 100K 위로 drift 한다. 이 절은 그 drift 를 막아 각 노드를 동작점에 앉히는 **클러스터 레이어 라우팅**을 다룬다.

> **스케일 표기.** 본 §7 은 **배포 128 동작점** (도출 기준 256 은 모든 값 2배 — OPERATING_POINT §1). 노드 풀 **134 = 2 μ-batch(124) + 잉여 10** · 프리필 60(OPERATING_POINT §4.1). 측정 = [cluster_balance.cpp](implementation/analysis/cluster_balance.cpp) `PREFILL=128 NODE_MAX=134`. **edge% 의 gate-shed 성분은 게이트 임계(100K+E) 의존이라 prefill-invariant**(총 edge 는 cap/풀 leftover 까지 더해 256 2.68% · 128 2.17%); on2·Σ편차 는 배치 62(256 의 123 절반)라 분산이 커 256 보다 헐렁(§6.8). 콜드스타트 후 배포 lifecycle 디코드 composition 100%(§6.8)로 마감.

> **PULS 코어와 독립.** 이 라우팅은 §6 의 배치 구성 알고리즘을 바꾸지 않는다 — 노드에 *어떤 요청을 보내는가* 만 정하는 위층이다. sim 은 PULS 와 독립이며 배치 구성 명중(개수·Σkv)만 보고 op-time 은 보지 않는다. 실 트래픽 부재로 가정 분포 **B**(short 20% [1–16K] / mid 70% [16–256K] / long 10% [256K–1M], 평균 ≈ 116K) 사용 — README 정직 disclosure 와 동일 선상.

### 7.1 동기: 센터링 없는 클러스터의 idle 폭발

이 로직 없이 클러스터를 돌리면 노드별 in-flight 가 글로벌 평균(116K)을 따라가 한 노드에 긴 요청이 쌓인다. 30M 캡 하에서 상주 평균이 길면 디코더가 적게 fit 한다: 배치를 크게(수백 규모) 짜 보면 **개수가 123 밑으로 떨어지고 Σkv 가 12.3M 위로 올라** 두 제어 타깃이 동시에 깨지고, 세 자원(PIM / GPU-A / FFN) 균형이 무너져 **idle 지수가 폭발**한다. 동작점은 *노드별* 100K 센터링에서만 성립하므로(§6.4 — 100K 는 평균을 *강제* 하는 값이 아니라 12.3M / 123 으로 캡을 유도하는 중간값), 클러스터 레이어가 각 노드 풀을 그 조건에 앉혀야 한다.

### 7.2 노드 HBM 의 실제 구성

라우팅의 타깃을 이해하려면 한 노드가 HBM(캡 15M, 배포 13.4M)을 어떻게 쓰는지부터 봐야 한다(평균 ctx 100K 기준):

```
HBM 캡 15M = 100K 기준 150 디코더 수용; 배포 풀:
  batch A 의 62 디코더 (6.15M, disjoint)
+ batch B 의 62 디코더 (6.15M, disjoint)
+ 잉여 10 디코더        (1.0M)
= 134 디코더 (13.4M) 상주  ← 캡 15M 미만(잉여를 작게 둬 대형 모델 가중치 여유, OPERATING_POINT §4.1)
```

2 배치가 "13.4M 을 다 쓴다"가 아니라 — **134 디코더 전체가 13.4M 을 채운다.** 2 배치는 그중 124 개를 *가리킬* 뿐, 잉여 10. (256 은 캡-충만 300 = 30M; 128 은 잉여를 10 으로 줄여 134 < 캡 150 — 그만큼 가중치 여유.)

**배치 만들기 = 메모리 할당 0.** warmup 후엔 새 배치가 안 생긴다. 활성 2 개 중 forward pass 끝난 *하나를 재구성* 할 뿐(`main_loop.py` `_recompose_mb`, §6.4-4):

```
batch A forward pass 끝남
  → A 의 옛 62 디코더가 풀로 돌아옴 (여전히 상주! 메모리 그대로)
  → candidate = (A 의 옛 62) + (잉여 10) = 72       ← 다 이미 상주
  → 그중 62 재선택 → 새 batch A
  → 메모리 할당 = 0
```

즉 노드 풀은 **134 디코더가 평균 100K** 로 상주하고, steering 이 매 iteration 그중 62 를 *골라* 배치를 만든다. 그래서 클러스터 라우팅의 노드별 타깃은 **count 124–134, 평균 ≈ 100K** — 이게 충족되면 steering 이 6.15M 짜리 배치 2 개를 disjoint 하게 뽑고, 재선택(72 → 62)도 성립한다.

> **134-평균이 아니라 62-배치가 동작점.** 노드 134-평균 100K 는 *용량/개수* 보장(캡 15M 안에 124–134 fit)이지 배치 구성을 직접 보장하진 않는다. 62-배치가 6.15M 명중하는 건 steering composer + 풀 다양성의 몫(§6.4 길이분산 무관). **배포 동작점 Σ편차 = 0.20%** (통합 lifecycle 의 live-KV 센터 composer, §6.8). 참고로 cluster_balance 의 *standalone* toxic-fit composer 는 더 헐렁(62-배치 평균 ~98,827·Σ편차 ~1.84% — live-KV 센터링 없이 분포만 보존)하나, 배포 healing 은 lifecycle 의 live-KV 센터링이라 **0.20%** 가 동작점값.

### 7.3 Cold-start: 엣지 게이팅 + interleave greedy

클러스터를 처음 채울 때:

1. **센터링·게이팅(엣지 격리).** 각 요청 길이 L 을 편차 d = L − 100K 로 본다. 도착 풀에서 *긴 것부터* 떼어 남은 평균이 100K + E 이하가 될 때까지 **엣지 노드**로 보낸다(기준은 *남은 평균* — `E` 는 토큰 단위 평균밴드 = Σd ≤ E × count; E = 1K → 평균 ≤ 101K, E = 0 이면 Σd → 0). 엣지 노드 = 초반의 *비정상(과도하게 긴)* 디코드를 전담하는 노드 — 격리해야 정상 노드 풀이 100K 로 센터된다. 떼어내는 비율 = **edge% = f(E) 단독**(분포 B 의 임계 위 꼬리 = P(x > V(E))). 노드 수·풀 크기와 무관, 분포에만 의존.
2. **interleave greedy 적재.** 남은 요청을 *도착 순(섞인 채)* 으로, 넣었을 때 그 노드 평균이 100K 에 가장 가까워지는 노드(min |추가후 mean − 100K|, 캡·count 한도 내)에 차례로 보낸다. 긴 거 + 짧은 거가 한 노드에 자연 interleave 되어 **count 134 · cap 15M · 평균 100K 가 동시에** 맞는다.

> **왜 크기순 정렬이 아니라 interleave 인가.** 134 × 100K = 13.4M(캡 15M)이라 세 조건(cap · count · 평균)을 동시에 만족하려면 한 노드에 긴 것과 짧은 것이 섞여야 한다. 편차 크기순으로 적재하면(KK / LPT 류) 긴 것부터 쌓여 **count 가 적은데 cap 이 먼저 차** 짧은 요청이 들어갈 자리를 잃는다(sim 측정: 배치 실패 다수 · on-point 붕괴). over/under swap 도 즉시 수렴해 이득이 없다. → 정밀 분할·swap 폐기, 단순 interleave greedy 채택.

### 7.4 Healing: 전략적 greedy refill (무축출)

운영 중 한 노드에서 요청이 **완료**되면(decode 종료, KV release) 그 자리가 빈다 — 메모리 churn 은 이 *완료* 순간뿐이고 full reservation·무축출(§6.4 풀 모델, no-eviction)이다. 빠진 뒤 노드는 두 결손을 갖는다:

- **개수 결손** `C_req = max(0, target − count)` — 2 μ-batch floor(124) 아래로는 안 떨어지게, 배포 풀까지(target = 134) 다시 채울 양.
- **편차 결손** `D_req = target_footprint − sum` (= 13.4M − sum) — 평균을 100K 로 되돌리려 새로 넣을 KV 의 총량. 긴 요청이 빠졌으면 `D_req` 큼(긴 게 필요), 짧은 게 빠졌으면 작음.

**힐링은 완료-단위(per-completion)다 — 이게 핵심.** 실서버 admission(§6.4)은 *한 요청이 완료될 때마다 즉시 한 개를 admit*한다(backpressure). 그러니 복구는 빠진 hole 하나하나에 대해 일어나고, 그 한 자리의 `ideal` 은:

```
한 요청(크기 hole) 완료 → count−1, sum−hole
ideal = (target_footprint − sum) / 남은 slot
      = hole               (slot = 1, 한 자리만 비었으므로)
→ 풀에서 ideal(≈hole) 에 가장 가까운 요청 1 개 admit  → 빠진 그 크기를 그대로 채움
```

즉 **빠진 크기를 같은 크기로 되채운다(like-for-like). 긴 요청이 빠지면 긴 요청이 들어온다** — 이것이 바로 Phase-1 의 "굵직한 구멍(독성) 우선 메우기(toxic-fit)"다. `ideal = hole` 이 hole 의 크기를 그대로 타깃하므로 큰 hole → 큰 admit, 작은 hole → 작은 admit 으로 자동. 그 결과:

- **상주 길이분포 보존** — 빠진 클래스를 같은 클래스로 채워 분포가 안 좁아진다(측정 128: 상주 긴요청(≥256K) 비율 8.23% → **7.36%** 유지).
- **각 길이 클래스를 도착률대로 소비** — 긴 요청이 도착률(≈7.6%)대로 정상 노드로 흘러 들어가 쌓이지 않음 → **엣지가 cold-start 비율(~2.2%)로 유지**.
- **inter-node swap 0** — 풀에서만 당긴다.

> **⚠ 한꺼번에(batched) 복구하면 toxic-fit 이 깨진다.** 여러 완료를 모아 한 번에 채우면 `ideal = D_req/slot` 이 *평균* 으로 뭉개져(예: 긴 1 + 짧은 다수 → ideal ≈ 100K) 긴 요청을 안 당긴다. 측정상(128) batched 는 상주 긴요청 8.04% → **0.01%** 로 완전히 굶기고(분포가 ~100K 로 좁아짐), 그만큼 긴 요청이 엣지로 쏠린다. **그래서 반드시 완료-단위(slot=1)로 복구** — 그러면 `ideal = hole` 이 되어 toxic-fit 이 성립한다. (per-completion 이 §6.4 의 실제 admission 메커니즘이기도 하다.)

> **Phase 2·3 과의 관계.** 원안 3-Phase 의 Phase 2(부호반대 *쌍*)·Phase 3(작은 원소 *다중 조합*)는 *유한·희소* 풀에서 hole 에 맞는 단일 원소가 없을 때 *조합* 으로 맞추는 일반화다. 무한 풀(§7)에선 best-of-K 가 hole 에 맞는 단일 원소를 거의 항상 찾아 **쌍·다중조합이 단일 pull 로 collapse** 한다. 반면 Phase 1(toxic-fit, 큰 것 우선)은 위처럼 `ideal = hole` 로 *그대로 살아있다*. 구현: `heal`([cluster_balance.cpp](implementation/analysis/cluster_balance.cpp) / [cluster_routing.py](implementation/src/cluster_routing.py)).

### 7.5 측정 결과 — E 스윕 · 안정성

가정 분포 B, Z = 256 노드, 캡 15M, on-point = compose(62, 6.15M ± 10%), 배포 풀 134(`PREFILL=128 NODE_MAX=134`).

**Cold-start E 스윕** (edge% = 엣지로 격리된 비율, on2% = disjoint 2-배치 명중률 = floor 124 의 진짜 의미):

| E | edge% | count ∈ 124–134 | \|평균 − 100K\| | on2% |
|---|---|---|---|---|
| 0 | 2.07 | 99.2% | 3.4K | 96.5 |
| **1K** | **2.17** | **94.9%** | **5.5K** | **93.8** |
| 5K | 1.73 | 89.5% | 7.5K | 88.7 |
| 10K | 1.53 | 77.7% | 12.7K | 73.4 |
| 20K | 0.89 | 63.3% | 19.7K | 57.0 |

E ↓ → 센터링 빡빡하나 엣지 ↑; E ↑ → 엣지 ↓ but 평균 drift → count floor 미달. **E = 1K 채택** — edge 2.17% 로 거의 완벽 센터링. cold-start 의 on2 < 100% 는 composition 실패가 아니라 일부 노드의 *count floor 미스*(~105 < 124)이며, 힐링이 메운다(62-배치라 256 의 ~98%보다 on2 다소 낮음 — §6.8; 그래도 배포 lifecycle 은 디코드 100%).

**Healing 안정성** (per-completion, 완료확률 p, 마지막 150 라운드 평균):

| p | count | ∈ 124–134 | \|평균 − 100K\| | on2% | 62-배치 평균 / Σ편차 |
|---|---|---|---|---|---|
| 1% | 133 | 98.0% | 4.7K | 94.4 | 98,823 토큰 / 1.84% |
| 3% | 133 | 97.7% | 5.1K | 93.9 | 98,828 토큰 / 1.89% |
| 5% | 133 | 96.1% | 5.3K | 92.9 | 98,827 토큰 / 1.84% |

힐링 진입 후 **drift 0**(warmup 직후 ≈ 마지막 라운드). per-completion 은 **cold-start 의 상태(다양·평균 100K)를 그대로 유지** — 노드 *평균*이 100K 에서 ±4.7\~5.3K(`|평균−100K|`, 센터링 품질), on2 ~93%, count ~133 정상(완료 자리만 채워 124–134 운용). 표의 `62-배치 Σ편차 1.84%` 는 **cluster_balance 의 standalone composer**(live-KV 센터링 없음) 값이고, **배포 동작점 Σdev = 0.20%** — 통합 lifecycle 의 live-KV 센터 composer(§6.8). on2 가 256(~98%)보다 낮은 것도 62-배치 분산(§6.8)이며, 배포 lifecycle 은 healing+steering 으로 **디코드 100% 명중·Σdev 0.20%**.

**toxic-fit 검증 — 긴 요청(≥256K) 보존** (E = 1K, p = 3%, 300 라운드):

| healing 방식 | 상주 긴요청%(cold) | 상주 긴요청%(late) | pull-긴요청% | on2% | \|평균−100K\| |
|---|---|---|---|---|---|
| batched (평균) | 8.04% | **0.01%** | 0.32% | 100.0 | 8 토큰 |
| **per-completion** | 8.23% | **7.36%** | 7.67% | 96.5 | 4.2K |

batched 는 긴 요청을 완전히 굶겨(8.04% → 0.01%) 분포를 ~100K 로 좁히고(그래서 dev 8·on2 100% 로 *과도하게* 깨끗) 긴 요청을 엣지로 쏠리게 한다. **per-completion 은 긴 요청을 보존**(8.23% → 7.36%, pull-긴요청 7.67% = 도착률대로 소비)해 toxic-fit 을 실현 → 엣지가 cold-start 비율 유지. **초반 ~2.2% 엣지 비용만 감수하면, 그 뒤로는 greedy cold-start + per-completion healing 으로 PIM 을 각 노드 평균 100K 동작점에서 무한정 운용**한다.

> **honest disclosure.** sim 가정: (a) 분포 B 는 가정값(실 트래픽 부재), (b) 무한 풀은 best-of-K 샘플로 모사, (c) churn 은 완료확률 p 의 추상화(실 decode-step 누적 아님), (d) steady-state 엣지는 직접 추적 대신 pull-긴요청률(≈ 도착률)로 간접 확인, (e) **본 cluster_balance sim 은 디코드 풀 센터링·on2 만** 본다(프리필→디코드 종속성·프리필 dual-target 미반영). 그 종속성·프리필(128 토큰 ∧ depth-work 12.8M)·**age-cap = 5** 를 다 넣은 통합 검증은 [cluster_lifecycle.cpp](implementation/analysis/cluster_lifecycle.cpp)(§6.8)이 **디코드·프리필 동시 100% 명중**으로 마감한다 — cluster_balance(분배·on2) + cluster_lifecycle(노드 생애)의 두-sim 분담. cold-start E-스윕엔 age-cap 영향이 작아 cluster_balance 에선 생략(§6.4 sweep).

---

## 8. Orthogonality to Complementary Techniques

### 8.1 Paged KV Memory Management

- **계층 구분:** Paged KV 관리 기법은 KV 캐시를 비연속 메모리 페이지로 관리하는 *메모리 관리* 계층이다. PULS PIM 은 HBM 에 상주하는 KV 데이터 위에서 attention 연산을 실행하는 *컴퓨팅 오프로드* 계층이다.
- **비간섭:** 양자는 서로 다른 추상화 수준에서 동작하며 인터페이스를 공유하지 않는다. PIM FSM 은 페이지 테이블 참조를 통해 비연속 KV 레이아웃을 투명하게 처리할 수 있으므로, 페이지 기반 물리적 배치 결정은 PIM 연산의 정확성에 영향을 주지 않는다.
- **이득 누적:** 페이지 관리가 제거하는 단편화 손실과 PIM 이 제거하는 GPU-side attention 비용은 독립적으로 누적된다.

### 8.2 Speculative Attention

- **기법 정의:** Speculative decoding 은 draft 토큰 생성 후 단일 forward pass 에서 병렬 검증을 수행한다. Speculative attention 은 이 검증 패스의 attention 비용을 최적화한다.
- **PIM 적용 가능성:** PULS PIM 은 토큰 출처 (draft / verified / speculation tree 내 위치) 에 무관하게 attention 연산을 동일하게 처리하므로, speculative attention 패스에 대해서도 PIM offload 가 성립한다.
- **이득 합산:** speculative decoding 이 forward pass 횟수를 줄이고, PIM 이 각 pass 의 attention 비용을 낮춘다. 두 최적화의 결합 처리량 향상은 곱셈적으로 작동한다.

### 8.3 Prefix KV Caching

- **기법 정의:** Prefix 공유 KV 캐시 히트로 prefill 연산 자체를 생략하는 기법군.
- **히트 구간:** PIM 의 prefill-side attention 부하도 비례하여 감소한다.
- **미스 및 decode 구간:** PIM offload 이득이 그대로 유지된다.
- **이득 누적:** Prefix caching 이 KV 재사용으로 전체 연산 규모를 축소하고, PULS 가 남은 연산의 attention 비용을 흡수한다. KV 히트율과 PIM offload 이득은 독립 변수로서 교차 항 없이 성능에 기여한다.

## 9. Open Empirical Work

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

본 architecture 문서의 정량 coverage:

- **Source decomposition** (Aux1·Aux2·F3·F5, η_HBM sensitivity sweep) — calibrated projection, [`README.md`](README.md#results) 참조
- **F1·F2 ablation, MFU plateau, admission ceiling** — 후속 calibration 으로 연기
- **절대 metric** (TTFT, TPOT, throughput) — silicon 부재로 영구 out of scope
