# PULS Architecture

**P**IM-**U**nified **L**LM **S**erving — scheduler-aware co-design.

- Motivation / problem statement / proposal overview — see [`README.md`](README.md)
- Quantitative source decomposition (Aux1·Aux2·F3·F5) — see [`README.md`](README.md#results)
- F1·F2 ablation + absolute metrics (TTFT / TPOT / throughput) — deferred to subsequent calibration / silicon absent

> **Generalization complete (2026-06).** The numbers here (deployed prefill 128 → decode 62 · 6.15M · ctx ≈ 100K, Instance A ≈ 3.20 TB) are the *canonical instantiation* for **Llama-3 70B + B200 + HBM4 16-high**. The *method* producing this operating point (three-resource balance derivation + steering · cold-start · healing · age-cap) is generalized over model/GPU and implemented as the C++ scheduler [`puls-engine/`](puls-engine/CONTRACT.md) (197 checks), deriving the operating point for any model · GPU spec within the HBM capacity limit. Fixed = HBM4 · SP-PIM · KV FP8; variable = model spec · GPU spec · prefill · die-stack · weight precision.

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
  - [5.1 Phase-aware Channel Activation](#51-phase-aware-channel-activation)
  - [5.2 Fixed-shape Handoff to Instance B](#52-fixed-shape-handoff-to-instance-b)
  - [5.3 PIM Overlap during Compute-Bound Windows](#53-pim-overlap-during-compute-bound-windows)
  - [5.4 Partial Resolution of Scheduling Predictability](#54-partial-resolution-of-scheduling-predictability)
  - [5.5 Prototype Vehicle: Self-authored Scheduler Framework](#55-prototype-vehicle-self-authored-scheduler-framework)
  - [5.6 Intra-instance Double-Buffering](#56-intra-instance-double-buffering)
  - [5.7 Acceleration Source Decomposition](#57-acceleration-source-decomposition)
- [6. Instance A Internal Scheduler Policy](#6-instance-a-internal-scheduler-policy)
  - [6.1 μ-batch Composition](#61-μ-batch-composition)
  - [6.2 Invariants](#62-invariants)
  - [6.3 Dispatch Policy: Event-driven + Dependency DAG](#63-dispatch-policy-event-driven--dependency-dag)
  - [6.4 Admission: The Operating Point (Pool Model)](#64-admission-the-operating-point-pool-model)
  - [6.5 Example Dispatch Trace](#65-example-dispatch-trace)
  - [6.6 Bound Analysis — Removed](#66-bound-analysis--removed-the-operating-point-is-balanced-no-single-bound)
  - [6.7 Implementation Requirements](#67-implementation-requirements)
  - [6.8 2-active μ-batch composition validation](#68-2-active-μ-batch-composition-validation)
- [7. Cluster Scale: Per-Node Pool 100K-Centering Routing](#7-cluster-scale-per-node-pool-100k-centering-routing)
  - [7.1 Motivation: Idle Blowup Without Centering](#71-motivation-idle-blowup-without-centering)
  - [7.2 What the Node's HBM Actually Holds](#72-what-the-nodes-hbm-actually-holds)
  - [7.3 Cold-start: Edge Gating + Interleave Greedy](#73-cold-start-edge-gating--interleave-greedy)
  - [7.4 Healing: Strategic Greedy Refill (No Eviction)](#74-healing-strategic-greedy-refill-no-eviction)
  - [7.5 Measurement — E Sweep · Stability](#75-measurement--e-sweep--stability)
  - [7.6 Cluster Scheduler C: Global Age-cap + Multi-turn Cache + Dependency-Contention TBT](#76-cluster-scheduler-c-global-age-cap--multi-turn-cache--dependency-contention-tbt)
- [8. Orthogonality to Complementary Techniques](#8-orthogonality-to-complementary-techniques)
  - [8.1 Paged KV Memory Management](#81-paged-kv-memory-management)
  - [8.2 Speculative Attention](#82-speculative-attention)
  - [8.3 Prefix KV Caching](#83-prefix-kv-caching)
- [9. Open Empirical Work](#9-open-empirical-work)

---

## 1. Key Observations

**O1.** In continuous-batched decode, the per-request variance in KV length induces variance in attention computation time, which manifests as straggler bubbles at the GPU batch granularity.

**O2.** All operations after attention (output projection, FFN) use the same weights and the same tensor shapes across requests. That is, the variable-length dependency is confined entirely to the attention stage.

**O3.** GPU HBM bandwidth utilization shows large variance by phase. During prefill's FFN and projection, the tensor cores are compute-bound and a substantial fraction of HBM bandwidth sits idle.

## 2. Design Principles

**P1. GPU Resource Non-intrusion.** PIM does not occupy or stall GPU compute resources.

**P2. Memory-bound Operations Only.** PIM handles only operations with low arithmetic intensity that are weight / activation streaming-bound (decode attention). Mounting compute-bound operations (projection, FFN, prefill attention) on PIM is unviable on both substrates — logic die and DRAM die:

- **Logic die placement limit** — The triple constraint of *area · thermal envelope · absence of SRAM* on the logic die keeps PIM compute density to single-digit percent of the GPU tensor core, so the acceleration contribution is negligible.
- **DRAM die placement limit** — (i) the MAC area encroaches on memory cells, reducing available KV cache capacity; (ii) DRAM thermal sensitivity causes throttling under sustained operation; (iii) non-standard DRAM circuit modifications are required → fab risk · yield burden.

Therefore PULS leaves compute-bound operations on the GPU tensor cores and restricts PIM strictly to the memory-bound regime.

**P3. Variable-Length Absorption.** PIM handles attention — the source of inter-request length variance — so that the GPU always receives fixed-shape tensors.

**P4. Scheduler Visibility.** The number of PIM-enabled channels and the operating phase are exposed dials that the serving scheduler can control.

**P5. TSV Occupancy and Compute-Bound Timing Alignment.**

- **External-bus contention from TSV / row buffer occupancy** — A PIM-activated channel occupies the TSV · row buffer, so it contends on the external HBM bus with the GPU-mode channels in the same stack.
- **Compute-bound timing activation** — PIM must be activated at timings when the GPU has entered a compute-bound region (during QKV / O projection or FFN execution) to avoid external bandwidth contention.

P2 (memory-bound operations only) ∩ P5 (compute-bound timing activation) = PULS's PIM dispatch policy — *process memory-bound operations, but only within the timing window in which a compute-bound operation is in flight.* This is the design rationale for §5.1 phase-aware channel activation and §5.3 overlap policy.

## 3. Architecture

### 3.1 Compute Substrate

A row-wise pipelined attention SFU is placed on the HBM4 **logic die (PHY)** — memory-die non-intrusion. Specifications:

- Head dim 128. **FP16 MAC core / FP8 (E4M3) KV-cache storage.** Weights and activations remain in BF16/FP16.
- **32-row tile FSM, 1.3 GHz clock.**
  - Tile data flow — the SFU reads a KV tile in FP8 from HBM, performs per-tile dequantization, and executes the FP16 MAC.
  - **FP8 KV regime** — tile TSV load time < PIM compute time → tile execution is in a *compute-bound* regime, and FSM deterministic timing holds.
  - **FP16 KV regime** — under the same mapping the tile switches to load-bound, taking roughly 2× the time.
  - The regime split is finalized by cycle-accurate measurement (Ramulator2-based).
- **Unified GEMM / GEMV handling** — Since it is a row-wise pipelined FSM, the control flow is identical whether the column width is 1 (GEMV, decode B=1) or a small matrix (GEMM, batch decode · multi-head). Only the per-tile MAC width differs; the FSM cycle structure is invariant.
- **No separate control unit — pure FSM operation.**
  - Directly reflects, at the substrate level, PIM's core advantage of eliminating instruction-dispatch / scheduling overhead.
  - Fixed cycle count per tile → execution time is deterministic → the scheduler can precisely predict PIM completion timing and plan overlap with GPU operations in advance.
- **Internal-path BW advantage (vs the GPU's external path).**
  - **PIM path (internal)** — uses only the row buffer → logic die SFU internally; no external-bus overhead → channel peak × 100% utilization.
  - **GPU path (external)** — row buffer → TSV → interposer → GPU memory controller → SM. Serialization delay · controller queuing · interposer latency cause loss against peak (η_HBM < 1, derived in the Discovery track).
  - **Result** — SP-PIM's aggregate effective BW over 2048 channels exceeds the GPU's aggregate effective BW by a factor of (1 / η_HBM). Substrate-level degrees of freedom are closed, so this is a near-fixed value (quantitative enters Aux2 / F3 — see [`README.md`](README.md#results)).

### 3.2 Channel-level PIM Toggle

Each of the 32 channels per HBM4 stack can be independently toggled between PIM and normal mode.

- **Scale** — 8 stacks × 32 channels per GPU = **256 channels** (per-GPU); aggregated across Instance A's 8 GPUs, **2048 channels** are independently toggleable.
- **Runtime split** — at any moment, k_total channels run PIM operations and the remainder serve GPU-side transactions.
- **Operational granularity = stack** — per-GPU k = n × 32, n ∈ {0, 1, …, 8}.
- **SP-PIM extremum** — activating k_total = 2048 across all of Instance A processes a single attention operation cooperatively across 8 GPUs in lock-step (see §3.4).

### 3.3 KV Cache Placement

The KV cache is sharded across Instance A's 8 GPUs.

**Inter-GPU sharding — not architecturally enforced.**

- **Options** — Head-sharded (GQA 8 KV heads ÷ 8 GPUs, 1 head/GPU), sequence-sharded (KV rows distributed + partial output reduce), and hybrid all arrive at identical attention results.
- **Default choice** — head-sharded (minimum communication cost, no output reduce required).
- **Scaling-compatible** — when scaling to TP > number of KV heads (e.g., TP=16 with GQA=8), it naturally transitions to sequence-sharded / hybrid.

**SP-PIM channel-level KV-row sharding — orthogonal to the inter-GPU sharding decision.**

- **Channel self-contained** — within each GPU's 256 channels, 32-way row-striped; 2048 channels in aggregate. Under any inter-GPU mapping, each channel sweeps its own KV row slice in a self-contained manner.
- **Invariance** — the same mapping repeats per layer; the KV remains permanently resident in Instance A's HBM (no inter-instance KV transfer).

### 3.4 Instance Disaggregation: Attention Block vs FFN Block

The transformer layer is split into an attention block and a post-attention block (FFN), each placed on a separate instance (for the per-layer 14-step flow and instance mapping, see [`instance_disaggregation.png`](figures/instance_disaggregation.png) in the README).

- **Inter-instance connection** — serially connected via an inter-instance pipeline.
- **Intra-instance parallelism** — Tensor Parallelism (TP).
- **Instance A additional** — PIM channel sequence parallelism (SP-PIM).

**Adopted configuration (Case A) — 16 GPUs total**

| Instance | GPUs | Internal Parallelism | Operations |
|---|---|---|---|
| A | 8 | TP=8 + SP-PIM (2048-channel Q-replicate) | input_layernorm, QKV projection, RoPE, KV save, attention (PIM), O projection |
| B | 8 | TP=8 | post_attention_layernorm, FFN gate/up/down, residual add |

**Instance A — Attention Block (GPU + PIM)**

- GPU configuration: 8 GPUs, TP=8. For inter-GPU KV sharding policy, see §3.3.
- The entire layer is processed jointly by the 8 GPUs (each GPU holds 1/8 of the weights per layer).
- PIM SFU: mounted on each GPU's HBM logic die. 8 stacks × 32 channels per GPU = 256 PIM channels. **2048 channels** in total across 8 GPUs.
- SP-PIM: processes a single attention operation cooperatively in lock-step across all 2048 channels of 8 GPUs. Q is broadcast and KV rows are sharded across channels to aggregate BW. Orthogonal to the inter-GPU KV sharding scheme (§3.3).
- KV cache: permanently resident in Instance A's HBM. No inter-instance transfer.

**Instance B — Post-Attention Block (non-PIM)**

- GPU configuration: 8 GPUs, TP=8. The same TP width as Instance A secures communication simplicity.
- Responsibilities: residual add + post_attention LN + FFN (gate, up, down) + residual add.
- Instance B GPU count: **8 adopted**. In the long-ctx regime, A_cycle (PIM attention + projection overlap) and B_cycle (FFN TP=n) naturally transition to A-bound, so further expansion of n yields limited per-GPU SLO goodput improvement.

**Instance B Memory Substrate — Hardware Cost Reduction**

Instance B's memory requirements differ structurally from Instance A's.

- **Access pattern.** Instance A: the KV cache accumulates in HBM proportionally to the number of requests × context length; both bandwidth and capacity demand SOTA HBM. Instance B: holds only per-layer FFN weights (FP16, gate + up + down GEMM weights). No KV accumulation; size is fixed.
- **Compute-bound determination.** FFN is compute-bound. Since HBM bandwidth is not the bottleneck of FFN processing time, HBM4's high bandwidth (≥ 4 TB/s) does not contribute to B_cycle.
- **Low-cost substrate substitution possibility.** In the regime where FFN weight streaming preserves the compute-bound characteristic, two options are available:
  - **(a) GDDR (GDDR6 / GDDR6X) substitution** — Substrate technology itself is changed. Capacity requirements (TP=8 basis) are also met by standard GDDR modules (24 GB). Substantial unit-cost reduction vs HBM4 8-stack (per-GB 3-5×) + additional savings on packaging cost (interposer + CoWoS). Trade-off: per-bit power rises 2-3× (long PCB path + higher clock + termination loss), partially mitigated by the low BW utilization of the compute-bound regime.
  - **(b) Low-stack-count HBM** — Same substrate technology, only the number of stacks is reduced (e.g., 8 stacks → 2-4 stacks). HBM's power efficiency (3-5 pJ/bit) is preserved. Trade-off: limited savings — part of the packaging cost is preserved.
  - Both options leave B_cycle unaffected (compute-bound regime preserved). The choice is a trade-off domain of cost / power / supply availability.
- **Comparison fairness preserved.** Since Instance B memory substrate changes (both options) do not affect B_cycle (compute-bound preserved), the PULS vs baseline comparison ratio is preserved. Quantitative analysis (required module count, cost / power ratio, sweet-spot stack count) deferred to subsequent calibration.

**TP+SP Selection Rationale (PP Rejected)**

**SP-PIM × PP incompatible.** SP-PIM distributes a single attention operation across all 2048 channels at time t. This requires all GPUs to participate in the same layer's attention in lock-step. PP, by definition, has each GPU processing a different layer at time t, which architecturally conflicts with SP-PIM. To leverage PIM's BW advantage, TP is forced.

**Inter-instance Data Transactions**

A and B are serially connected via an inter-instance pipeline. The order per layer is A → B → next layer A.

| Direction | Data | Notes |
|---|---|---|
| A → B | O projection output `[B × hidden]` | NVLink 4 SXM, intra-node send_recv |
| B → A | FFN output `[B × hidden]` (next layer input) | Same substrate |
| KV cache | **No transfer. Fixed in Instance A HBM** | — |

Two transfers per layer; 2L transfers per forward pass for an L-layer model. Asynchronous transfer allows hiding within A/B computation time.

**Pipeline Structure**

```
[μ-batch M]    A(layer i) ─ NVLink ─ B(layer i) ─ NVLink ─ A(layer i+1) ─ … ─ B(layer L)
[μ-batch M+1]  A(layer i-1) ─ … (steady-state overlap)
```

- A and B concurrently process different micro-batches (instance-level pipeline).
- Steady-state cycle = max(A_cycle, B_cycle).
- Pass through L layers = L × cycle.

Same design rationale as commercial deterministic compute scheduling cases (the PIM FSM's deterministic timing secures the predictability of pipeline scheduling).

### 3.5 Host↔PIM Interface (Interceptor)

**The essence of the Interceptor — a direct consequence of the attention-only scope.**

- **Mechanism — a MUX on the PHY DQ path** — The Interceptor only diverts the *destination* of the DQ output (normally bound for the GPU) to the PIM SRAM. The bank read sequence · timing is standard JEDEC RD / WR as-is — essentially the same operation as pulling data out to the GPU, with only the direction branched at the PHY stage.
- **Zero interface cost** — Since the scope is confined to the single decode-attention operation (P2), only 1 RFU bit of the JEDEC RD / WR command is occupied. No additional commands · decoders · queues · schedulers are required.
- **Contrast with existing HBM-PIM** — Existing architectures whose PIM scope extends to general-purpose GEMV cannot structurally reach this simplicity.

#### 3.5.1 Start (Interceptor)

The GPU permanently claims **1 RFU (Reserved For Future use) bit of the existing DRAM commands (RD/WR) as PIM_toggle**. For commands with PIM_toggle = 1, the **Interceptor (a MUX on the DQ output path)** on the logic die diverts the data destination to the PIM SRAM — the bank read itself remains a standard JEDEC RD/WR sequence as-is.

- **Layer-start metadata** — only `num_tiles` + `mode` (FP16/FP8) are additionally conveyed via RFU bits
- **Address-field reuse** — KV / Q / output addresses are naturally conveyed via the address field of the RD/WR commands the GPU issues anyway
- **Additional commands added = 0** — no new opcode definitions outside the JEDEC HBM4 standard command set

#### 3.5.2 End (Computed Wait)

- **FSM determinism** — fixed cycles per tile on the 1.3 GHz clock, jitter ±0 (see §3.1).
- **Computed wait** — At the moment the GPU fires PIM_toggle, the completion time is precomputed → the result is read from HBM exactly on time.
- **No separate synchronization mechanism required** — completion notification · interrupt · barrier all unnecessary.

Directly serves as the implementation basis for the *"PIM completion time precomputed"* assumption in §5.3 overlap policy and §6.3 scheduler dispatch.

#### 3.5.3 Result Delivery — via HBM

- **Only channel = HBM** — PIM resides on the HBM4 logic die, the GPU on a separate die; no direct P2P link.
- **Write → Read protocol** — PIM **writes** the result O to a designated address in HBM → the GPU **reads** that address exactly on time via computed wait.
- **Natural emergence of GPU-side conformance** — Data passing between GPU-internal kernels uses the same scheme (via global memory) → no separate DMA engine · doorbell mechanism required.
- **PIM-GPU TSV bandwidth contention**
  - **Shared vs separated paths** — PIM (internal HBM logic die) and Instance A GPU share the cell-array / TSV path; the external bus is GPU-only, since decode KV is processed in-place on the logic die and never leaves over the bus (path separation, §5.3).
  - **Two mitigation axes** — Contention, if any, therefore arises at the cell-array / TSV, and is expected to be mitigated on two axes: channel-independent toggling (§3.2) is intended to keep decode-KV / weight / prefill-KV on disjoint channels (space), and the GPU's compute-bound projection leaves the TSV idle for most of the window (time, §5.3).
  - **No time margin** — The op-time balance already yields t_pim ≤ t_gpu_a (OPERATING_POINT §2), so no time margin enters any sizing formula — the earlier 10% hedge (`PIM_SLACK_SAFETY_MARGIN`) was dropped.
  - **Open closure** — Quantitative TSV-saturation closure remains silicon-deferred.

## 4. Op Partitioning

| Operation | Phase | Executor | Rationale |
|---|---|---|---|
| Attention | **Decode** | **SP-PIM (Instance A)** | Memory-bound, KV streaming, absorbs variable length — aligned with P2 |
| Attention | **Prefill** | **GPU attention kernel (Instance A)** | Compute-bound, tensor core advantage; unsuitable due to PIM compute density deficiency |
| QKV / Output projection | All | GPU (Instance A) | Compute-bound, tensor core advantage; PIM mounting unrealistic due to logic die area constraints |
| FFN | All | GPU (Instance B) | Compute-bound; weight scale makes logic die area unrealistic |
| LayerNorm / Softmax / RoPE / Activation | All | GPU | Negligible compute, kernel launch overhead |

**Why decode attention specifically — Positive Fit Rationale.** For acceleration to hold on the PIM substrate, the following 3 conditions must be simultaneously satisfied:

1. **Online streaming feasible** — KV is not loaded all at once but processed by sweeping row-wise. The tile-level FSM allows the memory-path occupancy time to be deterministically partitioned (aligned with the 32-row tile FSM structure of §3.1).
2. **Reduction structure — O(1) internal state** — Intermediate results accumulate in O(1) (head_dim sized) inside the logic die SFU, avoiding logic die ↔ DRAM die round-trip traffic. The softmax denominator + row max accumulation is exactly this structure (the online softmax algorithm of FlashAttention).
3. **No need for massive MAC loading** — Not a reduction tree / dense matrix operation, but a narrow MAC sweep of size head_dim × tile_size suffices. Compatible with the logic die's area constraint (P1's KV-cache non-intrusion premise).

Only decode attention simultaneously satisfies these 3 conditions → the PIM scope is derived from the substrate's *positive fit*.

When the model/GPU changes, K1·K2 change so ctx_balance shifts — `puls-engine/core/derive.cpp` computes this balance for an arbitrary spec.

## 5. Scheduler Integration

### 5.1 Phase-aware Channel Activation

Instance A's SP-PIM aggregate channel count is fixed at k_total = 2048.

- **Always-on overlap** — PIM activates whenever decode-attn work exists, naturally overlapping with the HBM idle headroom of Instance A's GPU compute-bound stages (QKV · prefill_attn · O-proj) per O3 + §3.5.3.
- **No channel partitioning** — Because PIM is sequence-parallel across channels (§3.4), at any moment a single μ-batch's decode-attn occupies all 2048 channels — no channel-level partitioning across concurrent μ-batches is needed (Hermite identity on per-channel tile counts equates partition vs serialize).
- **Contention without a margin** — Residual TSV contention is expected to be absorbed by channel-independent toggling (§3.2) and path separation (§5.3) rather than a time margin — no fine-grained channel knob (TSV-saturation closure silicon-deferred).

Two bandwidth sources feed PIM here:

- **(i) The decode-KV channels themselves** — in GPU-only serving the GPU streams decode KV over the bus (memory-bound), but PULS processes it in-place on the logic die, so that traffic is structurally freed rather than merely overlapped.
- **(ii) The projection / prefill-KV channels** — being compute-bound, the GPU touches them for only a fraction of the window, leaving them idle the rest of the time.

→ **Source (i) is the larger effect** — decode is memory-bound under GPU-only serving.

- **Attention step** — In a mixed batch, prefill chunk tokens go to the GPU attention kernel and decode tokens to SP-PIM *concurrently*. With decode tokens present, all 2048 channels run a single lock-step op. For a pure-prefill batch (no decode rows), PIM op_time = 0.
- **Projection step (QKV / O-proj / FFN)** — No same-μ-batch PIM work. Under intra-instance double-buffering (§5.6), PIM processes the *next* μ-batch's decode-attn during the projection window — aligned with P5's compute-bound timing activation.

### 5.2 Fixed-shape Handoff to Instance B

Since PIM absorbs the KV-length dependency at the attention stage, the Instance A → Instance B inter-instance handoff tensor (§3.4) is always fixed-shape.

- Decode batch: B × hidden
- Prefill batch: (total prefill tokens) × hidden

- **Ragged chunks are fine** — Instance B's FFN GEMM depends only on the *total token count* of the batch, not on how those tokens are split across requests — so per-request prefill chunks may be ragged (the pool-model prefill steering distributes the 128 tokens unevenly to steer the depth-sum, §6.4).
- **The essence of the fixed-shape benefit** — the *attention*-stage KV-length variance has already been absorbed by PIM, so Instance B receives a shape that depends only on the token count; straggler bubbles from per-request KV-length variance are eliminated.
- **Scope caveat** — Instance A's GPUs still deal with the length dependency of prefill chunk attention, so they are not the direct beneficiary of this effect.

### 5.3 PIM Overlap during Compute-Bound Windows

Within Instance A, when the GPU projection is compute-bound, HBM bandwidth becomes partially idle. SP-PIM uses this headroom to overlap-process another micro-batch's decode attention (intra-instance double-buffering, §5.6).

- **Observation (premise):** In QKV/O projection and FFN compute-bound windows, HBM bandwidth utilization drops (see O3). This idle headroom is the region in which SP-PIM overlap is feasible.
- **Mechanism (means):** At GPU op entry, PIM channels are activated and the FSM completion time (deterministic cycle count) is precomputed to coordinate the GPU handoff timing. Channel-level independent toggling (§3.2) avoids conflict with the GPU command stream. Detailed behavior = §5.6.
- **Inter-instance pipeline alignment (effect):** Since the Instance A–B pipeline cycle (max(A_cycle, B_cycle)) becomes predictable via the PIM FSM's deterministic timing, the SP-PIM overlap window and the A↔B data-transfer timing can be precisely placed in micro-batch scheduling.

**SP-PIM Distribution Mechanism.**

- **Q-replicate / KV-row sharding** — Broadcast Q to all k_total channels, shard KV rows across channels → each channel independently sweeps its own KV slice (see §3.4).
- **Time derivation** — In both prefill chunk and decode batch scenarios, the number of tiles per channel is determined → tile count × tile time = SP-PIM attention time.
- **Ratio vs GPU baseline** — Determined by combining the internal-path BW advantage of §3.1 (exceeds by a factor of 1 / η_HBM) with ctx-dependent KV variance. **Quantitative derivation enters Aux2 / F3 — see [`README.md`](README.md#results).**

For quantitative evaluation of the concrete scheduling policy, see Open Empirical Work (§9 E6).

### 5.4 Partial Resolution of Scheduling Predictability

- **KV-length variance absorption:** Within a decode batch, the variance in per-request KV cache length induces variance in attention computation time, producing straggler bubbles (see O1). When PIM absorbs the variable-length attention, Instance B always receives fixed-shape tensors (see §5.2), eliminating this irregularity.
- **Removal of prefill-priority scheduling stalls:** In a mixed-batch environment, when prefill operations are prioritized, decode requests experience irregular delays. When PIM absorbs the length dependency of attention, the cause of prefill stalling decode is removed; prefill and decode coexist within the same batch, and the irregularity of decode delay is alleviated.

### 5.5 Prototype Vehicle: Self-authored Scheduler Framework

The scheduler core is implemented as a self-authored event-driven framework. OSS codebases (vLLM, Sarathi-Serve) serve only as references for baseline scheduler reimplementation; no code dependency.

- **Framework structure:** Self-contained data structure of event queue + dependency DAG + in-flight μ-batch window. Same invocation cadence as a production scheduler step. Attention calls are routed to the PIM executor, and layers split-dispatch across two instances — Instance A (attention + projection) ↔ Instance B (FFN) (§3.4).
- **Channel control:** Upon phase entry, the PIM channel count *k* is toggled at the scheduler step. Orthogonally compatible with the chunked-prefill policy.
- **TP=8 + SP-PIM integration:** SP-PIM Q-replicate is added on top of Instance A's GQA 8 KV head × TP=8 mapping. The attention kernel is implemented as an SP-PIM substitution.

### 5.6 Intra-instance Double-Buffering

Within the Instance A TP=8 cluster, GPU projection and SP-PIM attention concurrently occupy resources via μ-batch-level staggering.

- **Within the same μ-batch — serial enforced** — QKV projection (GPU) → attention (PIM SP) runs serially due to Q dependency. Inline overlap is impossible.
- **Across different μ-batches — no dependency** — While SP-PIM processes μ-batch M's attention, the GPU can pre-process μ-batch M+1's QKV projection. This asymmetry is the architectural basis for double-buffering.

**Sufficiency Conditions.**

- **Channel separation occupancy** — KV-cache channels (PIM mode, SP-PIM attention) and weight-streaming channels (GPU mode, projection) occupy non-overlapping subsets within HBM4's 2048 channels.
- **HBM bus contention avoidance** — Projection uses weight channels and SP-PIM attention uses KV channels independently → full overlap as long as channel separation holds.

**Expected Effects.**

- **Cycle shortening** — Instance A μ-batch processing time t_proj + t_attn → max(t_proj, t_attn).
- **Short-ctx** — projection-bound, PIM attention hidden.
- **Long-ctx** — PIM attention bound, projection hidden.

**Implementation requirement.** Instance A must hold the activation buffers of two micro-batches simultaneously, so the activation footprint grows 2×. This memory trade-off is to be quantified later (Open Work).

### 5.7 Acceleration Source Decomposition

PULS's overall acceleration contribution decomposes into five sources at op-level and systems-level.

| ID | Source | Mechanism | Primary Domain |
|---|---|---|---|
| F1 | SP-PIM attention | 2048-channel Q-replicate; a single attention cooperates in lock-step across 8 GPUs. GPU attention kernel → SP-PIM; the ratio is preserved at long-ctx | Layer processing time reduction, dominant at long-ctx |
| F2 | Projection ‖ PIM attention double-buffering | Concurrent execution of QKV/O projection (GPU) and SP-PIM attention across different micro-batches, A_cycle = max(t_proj, t_attn) | Instance A internal cycle shortening |
| F3 | Instance A–B inter-instance pipeline (PB1 eliminated) | In steady state, A (attention + proj) and B (FFN) concurrently execute different micro-batches. In a single-instance setup, attention → FFN is forced into serial processing | Effective time per layer t_A + t_B → max(t_A, t_B) |
| F4 | μ-batch staggering | Steady-state precondition for F2·F3 (not a standalone contribution item) | All regimes |
| F5 | PB3 elimination (channel-independent PIM scheduling) | SP-PIM channels operate independently of KV length, nullifying the max-KV straggler bubble within a batch. Trace-grounded variance (decomposed per axis): (i) public long-ctx agentic production trace + mid-ctx production chat trace, (ii) 1M-class real-doc benchmark dataset (alternative for regimes where a long-ctx production trace is absent). | SLO goodput, long-ctx production |
| (Aux) | Mixed batching resurrection | Weight sharing within the same batch of prefill + decode raises arithmetic intensity | TTFT / throughput trade-off |
| (Aux) | Bus traffic reduction | HBM-GPU bus transactions decrease when PIM processes attention. Large savings at long-ctx | Energy / cost |

## 6. Instance A Internal Scheduler Policy

Joint scheduling policy for GPU·PIM inside Instance A.

- **Dispatch location** — PULS attention dispatch is layered on top of the chunked-prefill + mixed-batch primitive (§5.5).
- **Weight sharing / attention split** — Within the same μ-batch, prefill·decode requests share QKV/O proj/FFN weights, while only attention is split by token-type (prefill chunk → GPU kernel / decode → SP-PIM) (§5.6).

**Core.** We reject fixed-slot scheduling and adopt *event-driven dispatch + 2-μ-batch lookahead*. Adaptive admission is the systems-level differentiator of PULS, and slot boundaries must not encroach on this degree of freedom.

### 6.1 μ-batch Composition

A μ-batch contains different requests in a phase mix. The KV cache is separated per request, and the weights (QKV proj, O proj, FFN) are shared by all tokens. Per-layer QKV proj and O proj run as a single bulk GEMM over the entire μ-batch; only attention branches by token-type (prefill chunk → GPU kernel / decode → SP-PIM).

### 6.2 Invariants

Whichever policy the scheduler adopts, the following 5 rules cannot be violated. I1·I2 are *correctness invariants* (data dependency); I3 is an *efficiency invariant* (split O-proj is mathematically equivalent, but is strictly never split due to weight reuse · MFU · kernel launch penalties); I4·I5 are *hardware resource constraints*.

| ID | Kind | Rule |
|---|---|---|
| I1 | correctness | prefill-attn(X) → dispatch only after QKV(X) completes (Q, K, V dependency) |
| I2 | correctness | decode-attn(X) → dispatch only after QKV(X) completes (PIM, FSM start condition) |
| I3 | **efficiency** | O-proj(X) → dispatch only after both prefill-attn(X) ∧ decode-attn(X) complete. Although row-wise independence permits splitting, the penalties of `W_O` 2× streaming + MFU drop + 2× kernel launch keep it always a single GEMM. Aligned with the production mixed-batch standard pattern |
| I4 | resource | At time t, the GPU resource runs only one GEMM / attention op (tensor core saturates; kernel concurrency ineffective) |
| I5 | resource | At time t, the PIM resource runs **only one** decode-attn op (SP-PIM occupies all 2048 channels). Not a head- or request-level constraint — within a single decode-attn op, multi-head · multi-request batching is free |

### 6.3 Dispatch Policy: Event-driven + Dependency DAG

**Definition.** Directed Acyclic Graph. Nodes are work units; an edge `A → B` denotes the precedence relation *"B can be dispatched once A completes."* Acyclicity precludes deadlock and guarantees topological order. PULS encodes the §6.2 invariants as DAG edges, reducing the dispatch decision to a *ready-node selection problem over a data structure*.

**Nodes.** For each μ-batch `X`, four work nodes: `QKV(X)`, `prefill-attn(X)`, `decode-attn(X)`, `O-proj(X)`.

**Edges.** I1·I2·I3 become precedence edges directly.

| Edge | Source invariant |
|---|---|
| `QKV(X) → prefill-attn(X)` | I1 |
| `QKV(X) → decode-attn(X)` | I2 |
| `prefill-attn(X) → O-proj(X)` | I3 |
| `decode-attn(X) → O-proj(X)` | I3 |

**Single μ-batch graph.**

```
              ┌─→ prefill-attn(M) ─┐
QKV(M) ──────┤                    ├──→ O-proj(M)
              └─→ decode-attn(M) ─┘
```

No explicit edges exist between distinct μ-batches — when resources (GPU·PIM) are available, arbitrary interleaving is possible. This is the graph-theoretic basis of look-ahead / back fill.

**Scheduler usage.** The scheduler maintains a DAG over the in-flight μ-batch window `{M_{i-1}, M_i, M_{i+1}}`, and at every kernel-completion event dispatches ready work from the two queues (GPU·PIM).

```
on event(kernel K of μ-batch X completes):
    update DAG: mark node K(X) as done
    refresh ready set: all unexecuted nodes whose precedences are done
    GPU_next = pick(ready GPU jobs,
                    priority: O-proj > prefill-attn > QKV,
                    tie-break: oldest μ-batch first)
    PIM_next = pick(ready PIM jobs,
                    tie-break: oldest μ-batch first)
    if GPU idle: dispatch GPU_next
    if PIM idle: dispatch PIM_next
```

**`pick` definition.** A priority-queue dequeue function. Arguments: (candidate set, priority rule, tie-break rule). Behavior: among candidates, narrow down to nodes belonging to the top-priority class of the priority rule, then pick a single node within that subset via the tie-break rule. Returns null if the candidate set is empty (in which case the resource sits idle).

**GPU priority order rationale.** Delaying O-proj is a *delay of the current μ-batch's completion* and thus directly impacts the critical path. QKV is the *next μ-batch's enabling* work, so it is prioritized only when there is a risk of PIM going idle.

**Sync guarantee.** Since I3 is auto-enforced as a DAG edge, there is no risk of missing synchronization. Owing to PIM FSM determinism, the completion time is computed at dispatch, allowing the next GPU dispatch to be pre-scheduled to the PIM completion moment.

**Natural emergence of look-ahead / back fill.** Since I2 is loose — *"any time after QKV completes"* — at moments when PIM is idle, decode-attn of another μ-batch can be started *regardless of which μ-batch the GPU is currently working on*. This emerges automatically from greedy dispatch without any separate policy specification.

### 6.4 Admission: The Operating Point (Pool Model)

The scheduler composes each μ-batch to drive the three resources (PIM = decode-attn, GPU-A = projection + prefill-attn, FFN = Instance B) to equal time, minimizing inter-instance and intra-instance idle. Composition is **three separate concerns**, each steered independently from its own in-flight pool.

1. **Admission = pool refill only.** `request_queue → in-flight (PREFILL)`, gated solely by the aggregate KV budget (`can_admit`). It does not look at the decode/prefill targets. A request with `prompt_len = 0` (decode-only), or one whose prefill is already complete, transitions straight to DECODE.
2. **Decode-set steering.** From the in-flight **DECODE pool**, select a set hitting two targets at once — **count 62 ∧ Σkv 6.15M** (deployed 128; the 256-derivation is 123·12.3M) — by local-greedy steering with an age-cap. Pure *selection* (no KV admission, no queue manipulation; KV is reserved at pool entry). Unselected requests age (`wait++`); selected reset (`wait = 0`).
3. **Prefill steering.** From the in-flight **PREFILL pool**, distribute **128 tokens** (the fixed FFN-batch knob, ① below) across members so the PREFILL_ATTN depth-sum hits **12.8M**, by the same per-token local-greedy + age-cap. A member receiving 0 tokens *stays in the pool* (it is not added as an empty chunk, which would inflate the μ-batch and starve decode) — a separate axis from decode.
   - **2-active prefill consistency** — *each* active μ-batch carries its own 128 prefill tokens (round total 256), steered independently; the two prefill sets are **request-disjoint** — chunk k+1 needs chunk k's KV, so a request prefills in at most one in-flight μ-batch (isomorphic to decode's shared-used exclusion).
   - **prefill_pool 80** — re-derived from 60: batch-1's greedy cherry-picks near-ideal depths, impoverishing batch-2's residual depth coverage; 80 is the knee (numbers in §6.8).

**These three concerns are re-run every iteration — per-iteration recomposition.**

- **Re-selection** — After every forward pass the μ-batch is re-selected from the pool (`_recompose_mb`), disjoint from other active μ-batches.
- **Window** — The in-flight window holds 2 active μ-batches (`_STAGGERING_TARGET_MB = 2`; capacity 3 = 2 active + 1 transition slack), which is the staggering precondition for F2/F3 overlap (§5.6, §6.5).
- **Restored freedom** — This restores the degree of freedom a sticky-cohort model destroys — composition tracks the pool as it drains and refills.

> **This is the real continuous-batching paradigm, not an approximation of it.** A production server holds a standing population mixing lifecycle stages — requests decoding (long generations in flight), requests prefilling (new prompts), and requests that just finished prefill and entered decode — and recomposes the batch every iteration (vLLM / Sarathi-Serve iteration-level scheduling). The pool model *is* that structure: an in-flight pool with PREFILL/DECODE states, KV resident in the pool, PREFILL→DECODE transition, admission refill, and per-iteration recomposition with mixed batching. The earlier sticky-cohort model — which froze a fixed cohort and processed it to completion together — was the *unfaithful* one; real servers never freeze a batch. The scheduling mechanism is therefore structurally faithful; what remains synthetic is only the *workload feed* (the warm-start seed and the abundant-pool assumption, honestly disclosed in the README), not the mechanism. So the operating-point composition validated in §6.8 holds on the real scheduling structure, not on a toy.

**The operating point (causal chain ① → ⑤).** The prefill token count fixes the FFN batch, which fixes the decode count, which with the balance ctx fixes the decode-KV sum:

| order | fixed value | value (prefill **128**, deployed; 256-derivation is 2×) | binding resource |
|---|---|---|---|
| ① | prefill tokens / batch | **128** (power-of-2, kernel-friendly; 256 is the derivation basis) | GPU-A (PREFILL_ATTN = Σ chunk×depth) |
| ② | balance time X (= throughput cycle) | **~25.5 µs** (X·L ≈ 2.0 ms = batch forward-pass period) | — |
| ③ | FFN batch | **190 tokens** | Instance B |
| ④ | **decode count N_dec (control target)** | **62** (= 190 − 128) | Instance B |
| ⑤ | **decode-KV sum (control target)** | **6.15M** | Instance A (PIM) |
| + | prefill KV-work (control target) | **12.8M** (= 128 × depth) | GPU-A |
| + | balance ctx | **~100K** (Llama70B+B200 balance value — specific to this model·chip pair; prefill-invariant, but re-derived by puls-engine if the model/GPU changes) | — |

The *count* (62) is independent of KV length (FFN sees only the token count); the *KV sum* (6.15M) is the sum of lengths (PIM sees only the sum). Both satisfied ⟺ mean ctx ≈ 100K. (Deployed 128; the 256-derivation is 123·12.3M, all 2× — OPERATING_POINT §1.)

**Local-greedy steering + age-cap (the `former` algorithm).** The control target is the pair *(count 62, Σkv 6.15M)* (deployed 128; the 256-derivation is 123·12.3M) — not a single average. Pure FIFO catches Σkv but misses the count on an off-average pool (measured spread 22–30%). So at each step the scheduler computes the *length it next needs* and admits the decoder closest to it (steering); a request that has waited `≥ AGE_CAP` is force-included (age-cap — fairness / FIFO intent). No global statistics, no future prediction — purely local:

```
one μ-batch (decode):                 # AGE_CAP = 5 (deployed/cluster; the old node-scheduler sweep was 2)
  n=0, S=0
  while n < target_count(62) and S < target_kv(6.15M) and pool:
    if any request with wait ≥ AGE_CAP: admit the oldest             # fairness (forced)
    else: admit decoder closest to ideal=(target_kv−S)/(target_count−n)   # steering
  remaining waiters: wait += 1
  → converges to (62, 6.15M); n increases monotonically → ≤62 steps.
prefill: per active μ-batch, distribute 128 tokens to depth-sum 12.8M by the same
  steering + age-cap (2×128/round, request-disjoint).
2 active μ-batch — re-composed on completion from (returned members + surplus) (capacity 3 = 2 active + 1 transition slack).
```

- **★ Length-distribution-agnostic (the key property).** On a heavily varied pool (real traffic), short + long requests are *combined* to hit both targets. Heavy / mixed / bimodal — any distribution works, because the average is never read; only the two targets are matched. Even an age-cap-forced off-size request is corrected by steering (a forced long request lowers `ideal` → the next picks several short ones), so the batch stays (62, 6.15M). A request that arrives first is processed within ≤ AGE_CAP+1 batches.
- **Operating parameters = target_count + target_kv + AGE_CAP.** Steering hits the target directly; `former` selects by nearest-to-`ideal`, not by any tolerance gate.
- **AGE_CAP trade-off (sweep).**
  - **Direction** — cap↑ → steering freedom↑ → spread↓, but waiting (latency)↑. cap↓ → FIFO-like → fair / low-latency but spread↑.
  - **Sweep** — `cap1: sp 3.1%` · `cap2: sp 1.2%, wait ≤3` · `**cap5: sp 0.7%, wait 5**` · `cap∞: sp 0.8% but starvation (wait 37)`. → **AGE_CAP = 5 adopted (deployed)**.
  - **Tail-bound knob** — the age-cap bounds the worst per-request token gap at exactly (cap+1) rounds (one batch cadence = a full forward pass X·L ≈ 2.03 ms, not the per-layer X).
  - **Measured ([PAUSE] KPI in csched)** — cap 5 → worst gap 6 rounds ≈ 12.2 ms (6× mean TBT), 3.4% of tokens paused, mean gap 1.12 — the average experience is unchanged.
  - **Looser caps** — trade +0.5% goodput for a 17× worse tail (cap 20 → 42.6 ms at request-p99, cap 100 → 205 ms stalls); this cost is invisible to batch-level TBT (identical p99 across the sweep) — hence the dedicated per-request KPI.
  - **Residual notes** — Spread 0.7% < cap2. The old node-scheduler was conservatively 2 (quantified in OPERATING_POINT §3).

**ctx 100K is the Llama70B+B200 balance value (specific to this model·chip pair), not an empirical guess.**

- **Re-derivable** — It is prefill-invariant, but if the model/GPU changes puls-engine re-derives this balance.
- **Closed form** — Solving the triple balance yields `ctx_balance = (K2+1)/K1` from the op-time coefficient ratios (PIM tile rate ÷ FFN flops/tok ÷ prefill-attn flops/tok·depth ÷ proj flops/tok); *prefill cancels out*, so the balance ctx is 100K for **every** prefill (§5 sweep B confirms).
- **Role** — to *derive* the targets (deployed 128: Σkv 6.15M = 62 × 100K; derivation 256: 12.3M = 123 × 100K), **not** to impose a mean on the workload.
- **Consequence** — This is why the algorithm is length-distribution-agnostic: it matches the two derived targets, however individual request lengths are distributed.

**prefill deployed 128 (derivation 256, vs 512).**

- **Scale knob, not balance** — prefill is not the balance ctx but the *scale knob* X — the smaller it is, the more the throughput cycle and HBM halve at zero TTFT / throughput cost (X is linear in prefill, so chunk · cycle cancel), provided the FFN batch stays above the MFU knee.
- **The halving ladder** — 512→256→**128**: cycle X 101→51→**25.5 µs**, HBM aggregate (decode) 60M→30M→**15M** = 9.8→4.92→**2.46 TB** (FP8 160 KiB/tok).
- **Capacity fit** — **Only 128 fits the 64 official stacks (4.096 TB)** (256·512 overflow — OPERATING_POINT §4.1). Instance A totals **3.20 TB** with the 2-active prefill's in-flight KV (pool 80) — above 12-high's 3.072 TB, so the deployed point is **16-high only**; the die-stack variable now actively binds.
- **Sole risk = FFN GEMM MFU saturation** — wave-quant estimation says batch ~128 saturates; the 128 deployment's batch = 62 + 128 = **190 (> knee, 48% margin)**, so it saturates but with less slack than 256 (379); the model fixes MFU = 0.6 so the knee is unobservable (silicon absent).
- **Decision** — **Deploy 128, fall back to 256 if measurement shows 190 insufficient** (512 batch 759 is safer still, vLLM-convergent).

### 6.5 Example Dispatch Trace

An instance of a **2-active-μ-batch + transition-tail** (capacity 3) in-flight window under PULS scheduler's balanced steady state (cycle balance maintained per §6.4; not *composing* 3 — 2 active overlap while the immediately-preceding batch drains as a tail). Example composition:

| μ-batch | Composition | Notes |
|---|---|---|
| P | {X: prefill chunk (✓ done before Init), G, H: decode} | The μ-batch immediately preceding M. By Init time, P's GPU stages have already completed and only decode-attn(G,H) remains on PIM — a tail state independent of P's original phase composition |
| M | {A: prefill chunk, B: decode, C: decode} | Current μ-batch |
| N | {D: prefill chunk, E: decode, F: decode} | Next μ-batch |

The table below is *one trace* of event-driven dispatch, not a fixed period. `T_i` is the time of a kernel-completion dispatch event; `Init` is the initial active state at the head of the trace.

| event | GPU work | PIM work | DAG state |
|---|---|---|---|
| Init | QKV(A,B,C) of M [back-fill: emergent GPU activity while PIM is busy on P] | decode-attn(G,H) of P [in progress] | O-proj(P) not ready (I3, PIM in progress); QKV(M) is the only ready GPU node → dispatched by priority dequeue |
| T1 | O-proj(P) [PIM(P) completion trigger] | decode-attn(B,C) of M | PIM(P) done → I3 satisfied → O-proj(P) fires; M QKV done → I2 satisfied → PIM dispatches M decode |
| T2 | prefill-attn(A) of M | (decode-attn(B,C) of M cont.) | GPU O-proj(P) done → priority pick: prefill-attn(M) (O-proj(M) still not ready, PIM busy on M) |
| T3 | QKV(D,E,F) of N [back-fill again] | (decode-attn(B,C) of M cont.) | GPU prefill(M) done → O-proj(M) still not ready → priority falls through to QKV(N) |
| T4 | O-proj(M) [PIM(M) completion trigger] | decode-attn(E,F) of N | PIM(M) done → I3 satisfied → O-proj(M); N QKV done → I2 satisfied → PIM dispatches N decode |
| T5 | prefill-attn(D) of N | (decode-attn(E,F) of N cont.) | Same pattern as T2: the M cycle's GPU pipeline repeats for N |

**Handling of G,H O-proj — GPU back-fill emergent property.**

- **No idle-wait** — Because I3 holds O-proj(P) not-ready while PIM is still processing P's decode-attn(G,H), the GPU does not idle-wait. By the priority dequeue (`O-proj > prefill > QKV`), it picks QKV(M) — the only ready node — and processes it as back-fill.
- **Trigger lands on a warm GPU** — Therefore at T1 the PIM-completion trigger fires *into a GPU already holding M's QKV complete*; the freed GPU resource then dispatches O-proj(P) on the trigger.
- **Emergent, not encoded** — This emergent GPU back-fill is the §6.3 priority dequeue realized — no explicit lookahead policy is encoded. The same pattern recurs at T3 (QKV(N) back-fill while PIM is on M).
- **Either ordering is handled** — If GPU's QKV(M) had been longer than PIM(P), T1 would shift to GPU QKV completion; under balanced admission this re-equilibrates within a few iterations, and the DAG handles either ordering automatically.

**Regime applicability.** The trace shape above is the steady-state attractor of PULS scheduler under balanced admission and is therefore **ctx-independent**: chunked prefill (§5.5) lets admission scale chunk granularity to maintain `t_PIM(decode-attn) ≈ sum of GPU stages` across any ctx within TBT SLO, so the same Init/T1–T5 dispatch ordering recovers regardless of chunk size. The trace ceases to apply only at very long ctx where balance is infeasible (§6.6 "A-bound natural transition") — a system-level property of multi-request schedulers under growing ctx, not a PULS-specific limitation.

### 6.6 Bound Analysis — Removed (the operating point is balanced, no single bound)

The legacy ctx-dependent bound analysis (short-ctx GPU-bound / long-ctx PIM-bound / mid-ctx B-bound) was a **pre-steering** framing. The §6.4 operating point steers the composition to (62, 6.15M) · (128, 12.8M), equalizing the three resource times → **there is no single binding resource** (idle ≈ 0 — §6.4 operating-point balance). "Which resource bounds the cycle" is therefore meaningless at the operating point, so this analysis is removed.

The sole exception = the **degenerate extreme (natural transition to A-bound).** When traffic is so long-ctx-only that the short requests to pair with are exhausted, PIM attention cannot hide inside the GPU-A window and crosses to A_cycle ≥ B_cycle — not present on real infinitely-varied traffic; a system-level limit of multi-request schedulers under growing ctx, not a PULS-specific one (OPERATING_POINT §6). (FP8 KV keeps the tile time compute-bound, halving t_decode-attn_PIM — substrate enabler in §3.1.)

### 6.7 Implementation Requirements

- Self-authored event-driven framework: 1 event queue, 1 dependency DAG, 2-active-μ-batch (+ transition tail, capacity 3) state in the in-flight window. Same invocation cadence as the production scheduler step.
- PIM completion-time predictor (FSM cycle-accurate).
- Idle fraction telemetry (per GPU-A · PIM · FFN, accumulated per measurement window).
- Pool-model composer (decode-set steering ‖ prefill steering ‖ admission refill, §6.4).

### 6.8 2-active μ-batch composition validation

**Multi-batch composition — 2 active μ-batch (deployed model).**

- **Window** — holds 2 active + 1 transition slack (capacity 3). When one batch's forward pass ends, it is **only *re-composed* from (returned members + surplus), never force-composing a 3rd.**
- **2-active prefill consistency** — both active μ-batches carry 128 prefill tokens each (round total 256), **request-disjoint** (chunk k+1 needs chunk k's KV — isomorphic to decode's shared-used exclusion); the lifecycle sim is corrected from once-per-round prefill to 2×128 disjoint (prefill→decode transitions 1286, ~1.9× the old 670).
- **prefill_pool re-derived 60 → 80** — batch-1's greedy consumes ~12–15 requests and cherry-picks near-ideal depths, so batch-2's residual depth coverage is impoverished (60: b2 96.3% / Σdev 1.84%). **80 is the knee** (b2 99.06% / 0.537%) inside the KPI-insensitive band; 100 rejected (unmeasurable composition gain for a measurable cache loss).
- **Verified composition** — The integrated lifecycle ([lifecycle.cpp](puls-engine/sim/lifecycle.cpp)) verifies that with the prefill→decode dependency and age-cap = 5, at deployed 128, **decode (62 ∧ Σkv 6.15M) hits 99.92% · Σdev 1.28% and prefill (2×128 ∧ depth-work 12.8M each) hits ≈99.5% combined · Σdev 0.32%, with no age-cap tail** (the logic is scale-invariant, prefill 256 ↔ 128 isomorphic). Repro: `puls_lifecycle 4000 64 5 2000` → decode 99.92% / Σdev 1.280% · prefill b1 100% / 0.107% · b2 99.06% / 0.537%.
- **Public validation** — `puls-engine` (`sim/lifecycle.cpp` · `validation/test_*.cpp`, 197 checks).

> **(2026-06 sim-faithfulness correction)** The earlier decode 100% / Σdev 0.38% came from a healing bug in the lifecycle sim (centering admit `ideal≈ctx_balance` collapsed the pool to all-mid, starving long requests to 0% → composition trivially perfect). With canonical healing (per-completion `ideal=hole`, like-for-like) + edge gating + prompt-independent realistic decode lengths + best-of-2000 infinite-pool emulation, the distribution-preserving (≈20/70/10) diverse pool gives decode ≈99.5% / Σdev ≈1.2% (age_cap 5) — consistent with the §6.4 / OPERATING_POINT §3 age-cap sweep cap5 spread (0.7%). Prefill stays 100% / Σdev ≈0.1%. (Since re-measured under the 2-active 2×128 disjoint prefill — current canonical above.)

## 7. Cluster Scale: Per-Node Pool 100K-Centering Routing

The §6 operating point (deployed 128: count 62 ∧ Σkv 6.15M; the 256-derivation is 123·12.3M) is hit by steering only when a node's in-flight pool is a **diverse pool centered at ~100K** (§6.4). On a single node admission maintains that pool, but at **server scale** (hundreds–thousands of nodes) the global arrival mean exceeds 100K, so per-node pools drift above 100K. This section covers the **cluster-layer routing** that prevents that drift and seats each node at the operating point.

> **Scale note.** This §7 is at the **deployed 128 operating point** (the 256-derivation is 2× every value — OPERATING_POINT §1). Node pool **149 = 124 + surplus 25** · prefill 80 (OPERATING_POINT §4.1). Measurements = [global_scheduler.cpp](puls-engine/core/global_scheduler.cpp) `PREFILL=128 NODE_MAX=134`. **The gate-shed part of edge% is prefill-invariant** (depends on the gate threshold 100K+E; total edge with leftover is 256 2.68% · 128 2.17%); on2 · Σ-dev are looser than 256 because the batch is 62 (half of 123), so variance is larger (§6.8). Cold-start is closed by the deployed lifecycle's decode composition 99.92% (§6.8).

> **Independent of the PULS core.** This routing does not change the §6 batch-composition algorithm — it is a layer above that only decides *which requests go to which node*. The sim is PULS-independent: it checks only batch-composition accuracy (count · Σkv), not op-time. With no real traffic available it uses an assumed distribution **B** (short 20% [1–16K] / mid 70% [16–256K] / long 10% [256K–1M], mean ≈ 116K) — on the same footing as the README's honest disclosure.

### 7.1 Motivation: Idle Blowup Without Centering

- **Drift** — Run a cluster without this logic and each node's in-flight pool tracks the global mean (116K), piling long requests onto a node.
- **Double target break** — Under the 15M cap, a high resident mean fits fewer decoders: compose batches large (hundreds-scale) and **count falls below 62 while Σkv rises above 6.15M** — both control targets break at once, the three-resource balance (PIM / GPU-A / FFN) collapses, and the **idle index blows up**.
- **Consequence** — The operating point holds only under *per-node* 100K centering (§6.4 — 100K is not a mean *forced* on the workload but the intermediate value 6.15M / 62 that derives the cap), so the cluster layer must seat each node's pool at that condition.

### 7.2 What the Node's HBM Actually Holds

To understand the routing target, first see how a node uses its HBM (cap 15M, deployed 14.9M; at mean ctx 100K):

```
HBM cap 15M = up to 150 decoders at 100K; deployed pool:
  batch A's 62 decoders (6.15M, disjoint)
+ batch B's 62 decoders (6.15M, disjoint)
+ 25 surplus decoders   (2.5M)
= 149 decoders (14.9M) resident  ← below the 15M cap (surplus kept small to free room for weights, OPERATING_POINT §4.1)
```

It is not that 2 batches "use up 14.9M" — **the full 149 decoders fill the 14.9M.** The 2 batches merely *point at* 124 of them, surplus 25. (256 was cap-full 300 = 30M; 128's surplus 25 so 149 < cap 150 — that much headroom for weights.)

**Forming a batch = zero memory allocation.** After warmup no new batch is created. Of the 2 active batches, the one whose forward pass finished is merely *recomposed* (`node_scheduler.cpp` (puls-engine sim) `_recompose_mb`, §6.4-4):

```
batch A's forward pass finishes
  → A's old 62 decoders return to the pool (still resident! memory unchanged)
  → candidate = (A's old 62) + (25 surplus) = 87       ← all already resident
  → reselect 62 of them → new batch A
  → memory allocation = 0
```

So the node pool keeps **149 decoders resident at mean 100K**, and steering *selects* 62 of them each iteration to form a batch. The per-node routing target is therefore **count 124–134, mean ≈ 100K** (the count band is the global-scheduler sim's `NODE_MAX = 134` — the pre-cache surplus-10 design; the deployed surplus 25 raises the resident pool to 149, §7.6) — once met, steering pulls two disjoint 6.15M batches and the reselection (87 → 62) also holds.

> **The operating point is the 62-batch, not the 149-mean.** A node's 149-mean of 100K guarantees *capacity/count* (fit 124–134 under the 15M cap), not batch composition directly. That a 62-batch hits 6.15M is the job of the steering composer + pool diversity (§6.4, length-variance-agnostic). **The deployed operating-point decode Σdev 1.28%** (the integrated lifecycle's per-completion ideal=hole composer, §6.8). This is in line with the global scheduler's *standalone* toxic-fit composer (puls-engine sim, 62-batch averages ~98,827 tokens, Σ-dev ~1.84% — distribution-preserving), since the faithful deployed healing preserves the same diverse pool; **1.28%** is the operating-point value (decode 99.92% hit).

### 7.3 Cold-start: Edge Gating + Interleave Greedy

Filling the cluster initially:

1. **Centering / gating (edge isolation).**
   - **Deviation view** — View each request length L as deviation d = L − 100K.
   - **Peel rule** — From the arrival pool, peel off the *longest first* to **edge nodes** until the remaining mean drops to ≤ 100K + E (the criterion is the *remaining mean* — `E` is a per-element mean band = Σd ≤ E × count; E = 1K → mean ≤ 101K, E = 0 → Σd → 0).
   - **Edge-node role** — Edge nodes serve the initial *abnormal (excessively long)* decodes — isolating them is what centers the normal-node pools at 100K.
   - **Peeled fraction** — **edge% = f(E) alone** (the tail above threshold V(E) of distribution B = P(x > V(E))). Independent of node count / pool size; depends only on the distribution.
2. **Interleave-greedy packing.** Place the remaining requests in *arrival order (shuffled)*, each into the node whose mean would land closest to 100K after insertion (min |post-add mean − 100K|, within cap / count limits). Long + short interleave naturally within a node, so **count 134 · cap 15M · mean 100K are satisfied simultaneously.**

> **Why interleave, not magnitude sort.** Since 134 × 100K = 13.4M (cap 15M), satisfying all three (cap · count · mean) at once requires long and short mixed within a node. Packing by deviation magnitude (KK / LPT style) stacks the long ones first, so **count is low while the cap fills first** and short requests lose their slots (sim: many placement failures · on-point collapse). Over/under swap also converges immediately for no gain. → precise partition / swap dropped, plain interleave greedy adopted.

### 7.4 Healing: Strategic Greedy Refill (No Eviction)

In operation, when a request on a node **completes** (decode ends, KV release) its slot frees — memory churn happens only at this *completion* instant, with full reservation and no eviction (§6.4 pool model). After the departures the node carries two deficits:

- **Count deficit** `C_req = max(0, target − count)` — the amount to refill back up to the deployed pool (target = 134), never letting it fall below the 2 μ-batch floor (124).
- **Deviation deficit** `D_req = target_footprint − sum` (= 13.4M − sum) — the total KV to inject to bring the mean back to 100K. Large `D_req` if a long request left (a long is needed), small if a short left.

**Healing is per-completion — this is the crux.** Real-server admission (§6.4) *admits one request the moment one completes* (backpressure). So recovery happens per freed hole, and that single slot's `ideal` is:

```
one request (size hole) completes → count−1, sum−hole
ideal = (target_footprint − sum) / remaining slots
      = hole               (slot = 1, only one seat freed)
→ admit the pool request closest to ideal (≈ hole)  → refill the same size that left
```

That is, **it refills the departed size with the same size (like-for-like). A long leaves, a long comes back** — exactly Phase-1's "fill the coarse (toxic) hole first." Because `ideal = hole` targets the hole's exact size: big hole → big admit, small hole → small admit, automatically. As a result:

- **Resident length distribution is preserved** — each departed class is refilled with the same class, so it does not narrow (measured at 128: resident long-request (≥256K) fraction 8.23% → **7.36%**, held).
- **Each length class consumed at its arrival rate** — longs flow into normal nodes at their arrival rate (≈7.6%) and never pile up → **edge stays at the cold-start rate (~2.2%)**.
- **Inter-node swap 0** — it pulls from the pool only.

> **⚠ Batched refill breaks toxic-fit.** If you pool many completions and refill at once, `ideal = D_req/slot` collapses to the *average* (e.g. one long + many short → ideal ≈ 100K) and never pulls a long. Measured (128), batched starves longs entirely (resident long 8.04% → **0.01%**, distribution narrows to ~100K) and shunts those longs to edge. **So always refill per-completion (slot=1)** — then `ideal = hole` and toxic-fit holds. (Per-completion is also the actual §6.4 admission mechanism.)

> **Relation to Phases 2 & 3.** The original Phase 2 (opposite-sign *pairs*) and Phase 3 (small-element *multi-combination*) generalize for a *finite/sparse* pool where no single element matches the hole; they *compose* one. The infinite pool (§7) makes best-of-K almost always find a single element at the hole, so **pairs/combinations collapse to a single pull**. Phase 1 (toxic-fit, biggest first), by contrast, stays alive exactly via `ideal = hole`. Implementation: `heal` ([global_scheduler.cpp](puls-engine/core/global_scheduler.cpp)).

Because the pool is effectively infinite, each admit nearly hits its `ideal`, so the mean sticks tightly to 100K. **Inter-node swap 0, edge 0** (it pulls only what it needs; the long requests not pulled are, by conservation, absorbed to edge at ≈ the cold-start rate).

### 7.5 Measurement — E Sweep · Stability

Assumed distribution B, Z = 256 nodes, cap 15M, on-point = compose(62, 6.15M), deployed pool 134 (`PREFILL=128 NODE_MAX=134`).

**Cold-start E sweep** (edge% = fraction isolated to edge, on2% = disjoint 2-batch hit rate = the real meaning of the 124 floor):

| E | edge% | count ∈ 124–134 | \|mean − 100K\| | on2% |
|---|---|---|---|---|
| 0 | 2.07 | 99.2% | 3.4K | 96.5 |
| **1K** | **2.17** | **94.9%** | **5.5K** | **93.8** |
| 5K | 1.73 | 89.5% | 7.5K | 88.7 |
| 10K | 1.53 | 77.7% | 12.7K | 73.4 |
| 20K | 0.89 | 63.3% | 19.7K | 57.0 |

- **Trade-off direction** — E ↓ → tighter centering but more edge; E ↑ → less edge but mean drifts → count floor missed.
- **Adoption** — **E = 1K adopted** — edge 2.17% for near-perfect centering.
- **on2 reading** — The cold-start on2 < 100% is not a composition failure but a *count-floor miss* on some nodes (~105 < 124), which healing fills (on2 is somewhat lower than 256's ~98% because the batch is 62 — §6.8; the deployed lifecycle still hits decode 99.92%).

**Healing stability** (per-completion, completion probability p, last 150 rounds averaged):

| p | count | ∈ 124–134 | \|mean − 100K\| | on2% | 62-batch mean / Σ-dev |
|---|---|---|---|---|---|
| 1% | 133 | 98.0% | 4.7K | 94.4 | 98,823 tokens / 1.84% |
| 3% | 133 | 97.7% | 5.1K | 93.9 | 98,828 tokens / 1.89% |
| 5% | 133 | 96.1% | 5.3K | 92.9 | 98,827 tokens / 1.84% |

- **Zero drift** — After healing engages, **drift is 0** (just-post-warmup ≈ last round).
- **State preservation** — Per-completion **preserves the cold-start state (diverse, mean 100K)** — the node *mean* within ±4.7–5.3K of 100K (`|mean−100K|`, centering quality), on2 ~93%, count ~133 (refill only the freed seats, within 124–134).
- **Which Σ-dev is which** — The table's `62-batch Σ-dev 1.84%` is **the global scheduler's standalone composer (puls-engine sim)** (distribution-preserving); the **deployed operating-point decode Σdev 1.28%** — the integrated lifecycle's per-completion ideal=hole composer (§6.8), in line with the standalone composer since both preserve the diverse pool.
- **on2 reading** — on2 below 256's ~98% is the 62-batch variance (§6.8) too; the deployed lifecycle hits **decode 99.92% · Σdev 1.28%** via healing + steering.

**Toxic-fit validation — long-request (≥256K) preservation** (E = 1K, p = 3%, 300 rounds):

| healing mode | resident long%(cold) | resident long%(late) | pull-long% | on2% | \|mean−100K\| |
|---|---|---|---|---|---|
| batched (avg) | 8.04% | **0.01%** | 0.32% | 100.0 | 8 tokens |
| **per-completion** | 8.23% | **7.36%** | 7.67% | 96.5 | 4.2K |

- **Batched fails** — Batched starves longs entirely (8.04% → 0.01%), narrowing the distribution to ~100K (hence the *too*-clean dev 8 · on2 100%) and shunting longs to edge.
- **Per-completion holds** — **Per-completion preserves longs** (8.23% → 7.36%; pull-long 7.67% = consumed at arrival rate), realizing toxic-fit → edge stays at the cold-start rate.
- **Bottom line** — **Pay only the ~2.2% initial edge cost, and thereafter greedy cold-start + per-completion healing run the PIM indefinitely at the per-node 100K operating point.** Public validation is `puls-engine` (`sim/lifecycle.cpp` · `validation/test_*.cpp`, 197 checks).

> **Honest disclosure.** Sim assumptions: (a) distribution B is assumed (no real traffic), (b) the infinite pool is emulated by best-of-K sampling, (c) churn is abstracted as completion probability p (not real decode-step accumulation), (d) steady-state edge is checked indirectly via the pull-long rate (≈ arrival rate) rather than tracked directly, (e) this **global-scheduler sim (puls-engine) covers only decode-pool centering / on2** (the prefill→decode dependency and prefill's dual target are not modeled here). That dependency, prefill (2×128 tokens ∧ depth-work 12.8M each, request-disjoint), and **age-cap = 5** are all included in the integrated validation by [lifecycle.cpp](puls-engine/sim/lifecycle.cpp) (§6.8), which closes with **decode 99.92% · prefill ≈99.5%** (post sim-faithfulness correction, §6.8) — the two-sim split of the global scheduler (distribution · on2) + lifecycle (node lifecycle). The age-cap effect on the cold-start E sweep is small, hence omitted in the global-scheduler sim (§6.4 sweep).

### 7.6 Cluster Scheduler C: Global Age-cap + Multi-turn Cache + Dependency-Contention TBT

§7.1–7.5 seat each node's decode pool at the operating point. The cluster scheduler **C** runs that exact node mechanism (resident surplus + per-iteration steering re-selection + per-completion healing, §6.4 / §7.4) and adds **two** cluster-level pieces, then measures real KPIs (TBT / TTFT / SLO goodput) under a dependency-and-contention TBT model.

**The two added pieces.**

| piece | what it does | impl |
|---|---|---|
| **Global age-cap** | A node computes `ideal = hole` (the completed request's size, like-for-like, §7.4). The global queue returns the request closest to that size — *unless* one has waited beyond the cap, in which case it force-routes the **oldest** (off-fit). Bounds cross-node waiting; a small cap → returns served fast → their KV is still inside the cache's evict window. | `pull_slot` |
| **On-node multi-turn KV cache** | 3-tier: **HBM hit** (resident, cost 0) → **SSD reload** (offloaded, cost ∝ length ÷ `offload_bw`) → **recompute** (gone). A completed request is HBM-cached iff `length > eligibility` (mid · long — the distribution-B short/mid boundary) **and** it fits the HBM left after the decode pool. `evict_age` idle → demote HBM→SSD; `gone_age` → recompute. The cache budget is **dynamically debited** by multi-turn pool-KV inflation (live pool KV beyond the design footprint, measured 0.242 TB ≈ 27% of the 0.891 TB budget — the old static accounting overstated hbmHit by ~9%p), and returns with an HBM-resident entry are **affinity-routed** to the holding node. | 3-tier `cache` |

**Division of labor — why all three.** surplus → composition (Σdev); global age-cap → latency / fairness *and* cache enablement (bounded wait ⇒ a returning session is served inside `evict_age` ⇒ HBM hit); cache → TTFT. They are coupled: the age-cap's forced off-fit injections would raise Σdev, but the **surplus absorbs them** (re-selection still hits 62 · 6.15M). Drop any one and C regresses toward a baseline. Surplus re-selection thus doubles as the absorber for **all** batch perturbations (age-cap forced injections, affinity mismatches), not just Σkv composition.

**Cache affinity + spill cap.** HBM-hit returns skip the global fit matching — they enter a per-holding-node affinity queue:

```
hole opens at node z
  priority 1: z's affinity queue non-empty → admit its longest-waiting member (regardless of length · cap_room)
  priority 2: queue empty → queue.pull_slot(hole) as usual
          (forced if a global aged exists, else nearest to ideal=hole)
```

- **Separate spill cap = 200 rounds (vs global age-cap 25).** A cached return's waiting is *rewarded* — ~2 ms/round of waiting avoids a 160–780-round SSD reload (~10× cheaper) — so it deserves its own bound. Sweep: spill 25→200 takes physical hits 67%→99.7% **and** improves Σdev 1.57%→1.38% (spilled requests became global forced off-fit injections — a worse perturbation than affinity's near-like-for-like). Even with no cap, waits self-bound at ~380 rounds (0.77 s ≪ TTFT SLO) because holes keep arriving (~17 rounds/node). Measured spill is near zero: **4 of 13,483 affinity candidates (0.03%)** — under the single global cap 25 it was 31% (4,317); the dedicated-cap separation cut it ~1,000×.
- **Why composition survives.** Affinity returns are former residents of that very node grown by ≤12K (new message + generation), and eligibility ≥16K excludes high-growth shorts → near-like-for-like by construction; the residual is absorbed by the existing surplus + re-selection + node age-cap machinery — re-confirming surplus re-selection as a *general-purpose perturbation absorber*.

**Dependency + contention TBT (the instance-dependency model).** This refines the §6.4 pool-time model and the §3.5 shared-path contention into the actual TBT:

```
TBT        = max(instance_a, t_ffn) × num_layers(80)
instance_a = max(t_pim, t_gpu_a) + β · max(0, t_pim − t_gpu_a)
```

- **Instance A→B dependency** — Instance A (PIM ‖ GPU-A) must finish before Instance B (FFN) → `max(instance_a, t_ffn)` (double-buffered steady-state throughput).
- **PIM↔GPU-A HBM contention** — contention-free iff `t_pim ≤ t_gpu_a` (PIM hides in GPU-A's shadow). When violated (PIM *exposed*), the exposed slice overlaps the next μ-batch's QKV back-fill on shared HBM → a `β · exposure` penalty. `β = 0` recovers the old `max(3) × L`; β is an **assumption label** (no silicon), swept below.
- **The operating point sits on the PIM-hiding edge.** `derive` balances `t_pim ≈ t_gpu_a`, so PIM is exposed *often but barely*: for C, 74.4% of μ-batches expose, yet only **expo 21 µs / μ-batch = 0.26 µs / layer ≈ 1%** of the 25.5 µs per-layer balance. That 1% (× β) is already baked into the TBT numbers below.

**Surplus ↔ cache HBM trade-off.** The surplus enlarges the decode pool's working set; the cache lives in what is left. **Surplus 25 (pool 149) is now the standard operating point; the U-knee (cache ON) is why** — surplus 10 / pool 134 was the pre-cache value. At surplus 25 there is more re-selection freedom (lower Σdev) without starving the cache. The cache budget itself is **0.891 TB** (was 1.075): the 2-active prefill's in-flight KV (pool 80 × ~56K tok ≈ +0.18 TB) now debits the pool footprint. Raising surplus further shrinks the cache (surplus 100 → HBM-hit 0); lowering the global age-cap below ~25 makes forcing explode (cap 5 → forced 11758 vs 4029, surplus can no longer absorb the off-fit). **(surplus 25, age-cap 25)** is where the two knees cross.

**A / B / C comparison** (8000 iters · Z = 64 · offload_bw 1e8; A = node-local baseline, no global age-cap, cache off · B = pure pre-positioning, no surplus · C = adopted):

| metric | A (baseline) | B (prepo) | **C (adopted)** |
|---|---|---|---|
| Σdev avg / worst | 1.42% / 16.5% | 20.1% / 94.1% | **1.42% / 22.6%** |
| TBT mean / p99 (µs) | 2061 / 2226 | 2598 / 3889 | **2059 / 2223** |
| TTFT mean / p99 (µs) | 0.94M / 8.9M | 0.89M / 9.0M | **0.78M / 8.96M** |
| SLO goodput (tok/s) | 3.78M | 2.15M | **3.79M** |
| TTFT-met % | 97.1 | 96.9 | **97.0** |
| cache HBM-hit % | 0 (off) | 91.1 | **87.9 (physical 99.7)** |
| PIM-exposed % | 76.2 | 82.5 | 74.4 |
| max wait (rounds) | 4551 (starve) | 70 | **26** |
| forced / poolMean | 0 / 117K | 26223 / 117K | 4029 / 116K |

*A historically ran surplus 10 (the pre-upgrade node design); shown here at the unified surplus 25 to isolate the global-age-cap + cache contribution. The surrounding sweeps (spill · β · eligibility) record the pre-2×-prefill accounting (prefill pool 60: hbmHit 91.1 · Σdev 1.38%); their verdicts — knee positions, β-robustness, orthogonality — carry over to the pool-80 canonical column.*

C is first or tied on every real KPI (TTFT-met 97.0 sits 0.1%p under A — the measured bill of pool 80 vs 60: hbmHit −3.2%p · TTFT +1.5% · goodput −0.3%, the honest charge for the previously unaccounted 2× prefill). **hbmHit 91.1 → 87.9 is corrected accounting, not a regression** — physical hit stays 99.7%. `poolMean` 116K and `forced` 4029 look high but are harmless — the surplus's re-selection still picks the low 62 (Σdev 1.42%), which is exactly why `poolMean` and Σdev decouple. B loses on TBT because its mis-composed batches expose PIM more and far wider (82.5%, expo 381 µs vs C's 21), where β bites.

**Honesty note — realistic SSD shrinks the gap.** With `offload_bw` corrected 2e7→1e8 (assumption labels below), the baseline A recovers substantially (TTFT 1.75M→0.94M, TTFT-met 97.1%). C's edge therefore rests on bounded waiting (max wait 26 vs A's 4551 — unbounded, length-biased starvation), composition, and the physically-real cache — smaller than under the old 2e7 constant, but more defensible.

**β sweep — C is β-robust, only B is sensitive:**

| β | C TBT mean / p99 | C SLO goodput | B TBT mean / p99 | B SLO goodput |
|---|---|---|---|---|
| 0 | 2047 / 2136 | 3.80M | 2407 / 3268 | 2.83M |
| 0.5 | 2058 / 2194 | **3.80M** | 2598 / 3889 | 2.15M |
| 1.0 | 2068 / 2253 | 3.80M | 2788 / 4510 | 1.83M |

C's goodput is flat across all β and its TBT moves only +21 µs (β 0→1) — exactly its `expo_us`, since the exposure β scales is ~1% / layer. B's goodput falls 36%. So β (set 0.5, conservative) changes only B's margin, never the C verdict.

**eligibility sweep — orthogonal to composition** (surplus 25):

| eligibility | cache HBM-hit % | savedR | **Σdev** |
|---|---|---|---|
| 0 | 97.2 | 25.0B | 1.41% |
| 16000 | 91.1 | 25.0B | **1.38%** |
| 64000 | 50.4 | 24.2B | 1.41% |

Σdev is invariant to eligibility (1.38–1.41%, noise-level) — it sets only *what* is cached, never the batch composition. At surplus 25 spare HBM is not the binding constraint (elig 0 → 97% hit, near-zero SSD), so **16000 is the principled, capacity-sparing choice** (HBM only for the mid / long that are expensive to recompute), re-confirmed at the locked operating point.

**Deferred / assumption labels.**
- **Event-driven dispatch DAG (§6.3) not implemented.** TBT here is the *analytic steady-state* form (double-buffering); the kernel-completion event timing of §6.3 is future work — the scheduler logic is reused, only the timing layer swaps.
- **Assumption labels (swept or fixed, not silicon-derived):** `β`, `offload_bw`, `think_gap`, the SLO thresholds, and the `max_tokens` distribution. `offload_bw` corrected 2e7→1e8: the old value sat 5% below the recompute break-even 2.1e7 B/round (= prefill 128 tok/round × 163,840 B/tok), making the SSD tier mathematically dominated; 1e8 ≈ a 50 GB/s NVMe array — still a swept assumption label. Absolute TBT / TTFT stay Llama70B + B200 `derive`-dependent.
- **Operating point unchanged.** surplus / global-age-cap / eligibility / β are cluster-layer knobs swept *on top of* the derived point (62 · 6.15M · ~100K · prefill 128), not changes to it.

Reproduction: `csched 8000 64 16000 200 25 300 0.5 1e8 5 25` (defaults affinity = on · dyncache = on · aff_spill = 200; new knobs argv[11] = affinity, argv[12] = dyncache, argv[13] = aff_spill) ([puls-engine/sim/csched.cpp](puls-engine/sim/csched.cpp) driver C · [puls-engine/scheduler/](puls-engine/scheduler/) queue · cache · [puls-engine/sim/kpi.h](puls-engine/sim/kpi.h) contention TBT). These cluster-layer pieces are absorbed into `puls-engine` (which owns the node mechanism, runtime path, and 197-check validation).

---

## 8. Orthogonality to Complementary Techniques

### 8.1 Paged KV Memory Management

- **Layer distinction:** Paged KV management is a *memory-management* layer that manages the KV cache as non-contiguous memory pages. PULS PIM is a *compute-offload* layer that executes attention operations over the KV data resident in HBM.
- **Non-interference:** The two operate at different abstraction levels and share no interface. Since the PIM FSM can transparently handle a non-contiguous KV layout via page-table reference, page-based physical placement decisions do not affect the correctness of the PIM operation.
- **Cumulative gains:** The fragmentation loss eliminated by page management and the GPU-side attention cost eliminated by PIM accumulate independently.

### 8.2 Speculative Attention

- **Technique definition:** Speculative decoding generates draft tokens and then performs parallel verification in a single forward pass. Speculative attention optimizes the attention cost of this verification pass.
- **PIM applicability:** Since PULS PIM processes the attention operation uniformly regardless of token origin (draft / verified / position within the speculation tree), PIM offload holds for the speculative attention pass as well.
- **Combined gains:** Speculative decoding reduces the number of forward passes, and PIM lowers the attention cost of each pass. The throughput improvement from combining the two optimizations works multiplicatively.

### 8.3 Prefix KV Caching

- **Technique definition:** A class of techniques that skip the prefill operation itself via shared-prefix KV cache hits.
- **Hit region:** The PIM's prefill-side attention load also decreases proportionally.
- **Miss and decode regions:** The PIM offload gain is preserved as is.
- **Cumulative gains:** Prefix caching shrinks the overall compute scale via KV reuse, and PULS absorbs the attention cost of the remaining computation. The KV hit rate and the PIM offload gain are independent variables that contribute to performance with no cross term.

## 9. Open Empirical Work

*Common simulation assumption: all requests are treated as new requests entering for the first time without a KV cache hit (no prefix duplication). In real workloads where KV hits occur, the prefill-side attention load decreases further, so PULS's relative improvement is expected to exceed the present simulation results.*

**E1.** SP-PIM aggregate (2048 channels) vs H100 GPU attention kernel token throughput comparison.

- **PIM token/s** — derived from the FSM specification (FP8 KV premise, §3.1).
- **GPU baseline** — measured token/s of the H100 attention kernel (same FP8 KV mode).
- **HBM4 estimate** — H100 HBM3 measurement × peak-ratio scaling.

**E2.** Cycle-accurate measurement of H100 HBM bandwidth utilization curves by LLM serving phase.

- **Output** — quantification of the PIM-available bandwidth headroom.
- **HBM4 scaling** — H100 measurement → HBM4 estimate.

**E3.** Sensitivity sweep of the phase-aware channel split ratio and derivation of the optimal policy.

**E4.** Tuning of mixed-batch chunk size and quantitative evaluation of arithmetic intensity improvement.

**E5.** Decode-batch straggler bubble analysis.

- **Cause** — per-request KV length variance induces attention-time variance → intra-batch bubbles.
- **Measurement** — bubble-size quantification (as a function of KV-length variance).
- **Comparison** — bubble fraction and overall throughput loss, before vs after PIM application.

**E6.** Measurement of compute-bound window (FFN, prefill-side projection) headroom in the production-grade scheduler.

- **Occupancy ratio** — share of compute-bound windows in total compute time.
- **HBM bandwidth utilization** — concurrent extraction of GPU HBM utilization in that window.
- **Usage** — grounds for the available-headroom of the PIM overlap hypothesis (§5.3).

**E7.** Quantitative analysis of HBM-GPU bus-transaction reduction by PIM offload.

- **Measurement items** — GPU-side HBM transaction count + total transfer volume (before vs after PIM application).
- **Phase decomposition** — separate measurement of transaction savings by prefill / decode.
- **Inter-instance cost comparison** — comparison with the cost of `[B × hidden]` transfer between Instance A/B.

---

Quantitative coverage of this architecture document:

- **Source decomposition** (Aux1·Aux2·F3·F5, η_HBM sensitivity sweep) — calibrated projection, see [`README.md`](README.md#results)
- **F1·F2 ablation, MFU plateau, admission ceiling** — deferred to subsequent calibration
- **Absolute metrics** (TTFT, TPOT, throughput) — silicon absent, permanently out of scope
