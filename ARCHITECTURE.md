# PULS Architecture

**P**IM-**U**nified **L**LM **S**erving — scheduler-aware co-design.

- Motivation / problem statement / proposal overview — see [`README.md`](README.md)
- Quantitative evaluation (acceleration multiples, latency / throughput absolute values) — to be measured in the Phase 3 calibration track

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
  - [5.3 PIM Overlap during Compute-Bound Windows](#53-pim-overlap-during-compute-bound-windows)
  - [5.4 Partial Resolution of Scheduling Predictability](#54-partial-resolution-of-scheduling-predictability)
  - [5.5 Prototype Vehicle: Production Serving Stack Fork](#55-prototype-vehicle-production-serving-stack-fork)
  - [5.6 Intra-instance Double-Buffering](#56-intra-instance-double-buffering)
  - [5.7 Acceleration Source Decomposition](#57-acceleration-source-decomposition)
- [6. Instance A Internal Scheduler Policy](#6-instance-a-internal-scheduler-policy)
  - [6.1 μ-batch Composition](#61-μ-batch-composition)
  - [6.2 Invariants](#62-invariants)
  - [6.3 Dispatch Policy: Event-driven + Dependency DAG](#63-dispatch-policy-event-driven--dependency-dag)
  - [6.4 Adaptive Admission](#64-adaptive-admission)
  - [6.5 Example Dispatch Trace](#65-example-dispatch-trace)
  - [6.6 Bound Analysis](#66-bound-analysis)
  - [6.7 Implementation Requirements](#67-implementation-requirements)
- [7. Orthogonality to Complementary Techniques](#7-orthogonality-to-complementary-techniques)
  - [7.1 Paged KV Memory Management](#71-paged-kv-memory-management)
  - [7.2 Speculative Attention](#72-speculative-attention)
  - [7.3 Prefix KV Caching](#73-prefix-kv-caching)
- [8. Open Empirical Work](#8-open-empirical-work)

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

P2 (memory-bound operations only) ∩ P5 (compute-bound timing activation) = PULS's PIM dispatch policy — *process memory-bound operations, but only within the timing window in which a compute-bound operation is in flight.* This is the design rationale for §5.1 phase-aware channel split and §5.3 overlap policy.

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
  - **GPU path (external)** — row buffer → TSV → interposer → GPU memory controller → SM. Serialization delay · controller queuing · interposer latency cause loss against peak (η_HBM < 1, derived in Phase 0 Discovery).
  - **Result** — SP-PIM's aggregate effective BW over 2048 channels exceeds the GPU's aggregate effective BW by a factor of (1 / η_HBM). Substrate-level degrees of freedom are closed, so this is a near-fixed value (quantitative = disclosed after Phase 3).

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
- **Comparison fairness preserved.** Since Instance B memory substrate changes (both options) do not affect B_cycle (compute-bound preserved), the PULS vs baseline comparison ratio is preserved. Quantitative analysis (required module count, cost / power ratio, sweet-spot stack count) belongs to Phase 3.

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

## 5. Scheduler Integration

### 5.1 Phase-aware Channel Split

At the entry of each μ-batch step, the scheduler determines the SP-PIM aggregate channel count k_total. Max k_total = 2048 across Instance A's 8 GPUs.

- **Attention step** — In a mixed batch, prefill chunk tokens are handled by the GPU attention kernel and decode tokens by SP-PIM *concurrently*. If the batch contains decode tokens, k_total = 2048 is activated (a single attention op cooperates in lock-step over 2048 channels). For a pure-prefill batch, k_total = 0.
- **Projection step (QKV / O-proj / FFN)** — No PIM work in the same μ-batch. However, under intra-instance double-buffering (§5.6) in which PIM pre-processes the next μ-batch's decode attention, k_total is activated during the projection window — aligned with P5's *compute-bound timing activation* principle. k_total is not binary but a continuous stack-granular dial (per-GPU n × 32, n ∈ {0..8}) — in regions where projection is fully compute-bound, k_total = 2048 (upper bound); when MFU saturation falls short or additional GPU-side BW is required, it is partially withdrawn (P4).

This split can be changed at step entry via a single toggle command, and due to channel-level independence does not conflict with the GPU-side command stream.

### 5.2 Fixed-shape Handoff to Instance B

Since PIM absorbs the KV-length dependency at the attention stage, the Instance A → Instance B inter-instance handoff tensor (§3.4) is always fixed-shape.

- Decode batch: B × hidden
- Uniform-chunk prefill batch: (B · chunk) × hidden

Instance B's GPUs perform only uniform FFN GEMMs without ragged batching handling, so intra-batch straggler bubbles are eliminated. (Instance A's GPUs still deal with the length dependency of prefill chunk attention, so they are not the direct beneficiary of this effect.)

### 5.3 PIM Overlap during Compute-Bound Windows

Within Instance A, when the GPU projection is compute-bound, HBM bandwidth becomes partially idle. SP-PIM uses this headroom to overlap-process another micro-batch's decode attention (intra-instance double-buffering, §5.6).

- **Observation (premise):** In QKV/O projection and FFN compute-bound windows, HBM bandwidth utilization drops (see O3). This idle headroom is the region in which SP-PIM overlap is feasible.
- **Mechanism (means):** At GPU op entry, PIM channels are activated and the FSM completion time (deterministic cycle count) is precomputed to coordinate the GPU handoff timing. Channel-level independent toggling (§3.2) avoids conflict with the GPU command stream. Detailed behavior = §5.6.
- **Inter-instance pipeline alignment (effect):** Since the Instance A–B pipeline cycle (max(A_cycle, B_cycle)) becomes predictable via the PIM FSM's deterministic timing, the SP-PIM overlap window and the A↔B data-transfer timing can be precisely placed in micro-batch scheduling.

**SP-PIM Distribution Mechanism.**

- **Q-replicate / KV-row sharding** — Broadcast Q to all k_total channels, shard KV rows across channels → each channel independently sweeps its own KV slice (see §3.4).
- **Time derivation** — In both prefill chunk and decode batch scenarios, the number of tiles per channel is determined → tile count × tile time = SP-PIM attention time.
- **Ratio vs GPU baseline** — Determined by combining the internal-path BW advantage of §3.1 (exceeds by a factor of 1 / η_HBM) with ctx-dependent KV variance. **Quantitative derivation will be disclosed after the Phase 3 sim closes.**

For quantitative evaluation of the concrete scheduling policy, see Open Empirical Work (§8 E6).

### 5.4 Partial Resolution of Scheduling Predictability

- **KV-length variance absorption:** Within a decode batch, the variance in per-request KV cache length induces variance in attention computation time, producing straggler bubbles (see O1). When PIM absorbs the variable-length attention, Instance B always receives fixed-shape tensors (see §5.2), eliminating this irregularity.
- **Removal of prefill-priority scheduling stalls:** In a mixed-batch environment, when prefill operations are prioritized, decode requests experience irregular delays. When PIM absorbs the length dependency of attention, the cause of prefill stalling decode is removed; prefill and decode coexist within the same batch, and the irregularity of decode delay is alleviated.

### 5.5 Prototype Vehicle: Production Serving Stack Fork

We fork the **chunked-prefill + mixed-batch OSS scheduler** codebase and insert a PULS dispatch hook to realize a prototype of our scheduler policy (§6).

- **Hook location:** scheduler worker / model runner boundary. Attention calls are routed to the PIM executor, and the layer is dispatched split across two instances — Instance A (attention + projection) ↔ Instance B (FFN) (§3.4).
- **Channel control:** Upon phase entry, the PIM channel count *k* is toggled at the scheduler step. Orthogonally compatible with the chunked-prefill policy.
- **TP=8 + SP-PIM integration:** SP-PIM Q-replicate is added on top of Instance A's GQA 8 KV head × TP=8 mapping. The existing TP code path is reused; SP-PIM is implemented as an attention-kernel substitution.

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

### 6.4 Adaptive Admission

The scheduler dynamically adjusts the per-μ-batch composition decision *on a per-iteration basis*. The GPU/PIM idle fractions of the previous iteration are measured to regulate the next μ-batch's admission. Hooked on top of the chunked-prefill policy as dynamic adjustment of chunk size · decode batch (§5.5).

**Decision Rule.** The essential frame of admission control:

- **Layer 1 — μ-batch composition** (on top of the chunked-prefill + mixed-batching primitive): decides the prefill-chunk vs decode token mix and N. **The determining factor of TTFT / TBT SLO condenses to this layer.**
- **Layer 2 — DAG dispatch** (§6.3): Automatic ready-node selection on top of the Layer 1 result. Since processing is serial, it cancels with admission variables — adaptive degrees of freedom concentrate in Layer 1.

Adaptive admission's primary objective = balancing the two instances of the inter-instance pipeline cycle `max(A_cycle, B_cycle)` (both fully utilized). Secondary objective = balance of GPU·PIM double-buffering inside Instance A (§5.6). A hysteresis deadband suppresses oscillation from GPU jitter · workload variance (see the Deadband Policy section).

| Layer | Measurement | Diagnosis | Admission adjustment |
|---|---|---|---|
| Inter-AB (primary) | `A_cycle > B_cycle` (B idle) | A-bound (long-ctx) | admission ↓ effect limited (the PIM attention component of `A_cycle` depends on KV length) — B idle naturally accepted |
| Inter-AB (primary) | `A_cycle < B_cycle` (A idle) | B-bound (short-ctx + low batch) | admit prefill chunk → `A_cycle` increases, balance restored |
| Intra-A (secondary) | GPU idle > `θ_high`, PIM busy | PIM-dominant inside Instance A | admit additional decode → fill PIM window |
| Intra-A (secondary) | PIM idle > `θ_high`, GPU busy | GPU-dominant inside Instance A | admit prefill chunk → fill GPU window |
| — | Both layers below `θ_low` | balanced | maintain current admission |
| — | Both layers idle | underloaded | enlarge μ-batch size or accelerate wait tokens |

**Deadband Policy: Ctx-tiered Static Lookup.**

- **Width formula** — Deadband width = `2σ_total` (control-theory standard, hysteresis stability condition).
- **`σ_total` decomposition** — RSS sum of GPU jitter (L2 hit rate / warp scheduler / HBM controller queuing / kernel launch) and workload variance (KV length variance / arrival jitter).
- **Rationale for ctx-tiered adoption** — The longer the ctx, the longer the cycle, the greater the accumulated `σ`, and the more dominant the KV variance influence → static per-ctx lookup.

| ctx | σ_total estimate (qualitative) | deadband width |
|---|---|---|
| Short-ctx (2k–8k) | low (GPU-jitter dominated) | narrow |
| Mid-ctx (~32k) | medium | medium |
| Long-ctx (128k–1M) | high (KV-variance dominated) | wide (enters clamp 0% region) |

Quantification of `σ_total`, deadband sweep, and the online adaptive variant (a per-iteration `σ` estimator that auto-updates the width) are all future work outside the scope of this study — since the scheduler simulator lacks a real-hardware jitter model, the very definition of σ measurement is absent. This evaluation measures only the qualitative behavior of the dispatch policy in the regime where the GPU·PIM cycle is balanced (balanced regime).

**Admission Lower Bound: MFU Floor.** `N ≥ N_sat` (FFN GEMM saturating knee) — below this, GEMM MFU is sub-saturating and kernel-launch overhead dominates. The upper bound belongs to the TPOT SLO model domain (future work). Per-ctx binding:

| ctx regime | Binding |
|---|---|
| 2k–256k | B-bound (FFN GEMM saturating knee) |
| 256k–512k | Transition |
| ≥ 512k | A-bound (B latency hidden inside A_cycle) |

### 6.5 Example Dispatch Trace

An instance of a 3-μ-batch in-flight window under PULS scheduler's balanced steady state (cycle balance maintained per §6.4). Example composition:

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

**Handling of G,H O-proj — GPU back-fill emergent property.** Because I3 holds O-proj(P) not-ready while PIM is still processing P's decode-attn(G,H), the GPU does not idle-wait. By the priority dequeue (`O-proj > prefill > QKV`), it picks QKV(M) — the only ready node — and processes it as back-fill. Therefore at T1 the PIM-completion trigger fires *into a GPU already holding M's QKV complete*; the freed GPU resource then dispatches O-proj(P) on the trigger. This emergent GPU back-fill is the §6.3 priority dequeue realized — no explicit lookahead policy is encoded. The same pattern recurs at T3 (QKV(N) back-fill while PIM is on M). If GPU's QKV(M) had been longer than PIM(P), T1 would shift to GPU QKV completion; under balanced admission this re-equilibrates within a few iterations, and the DAG handles either ordering automatically.

**Regime applicability.** The trace shape above is the steady-state attractor of PULS scheduler under balanced admission and is therefore **ctx-independent**: chunked prefill (§5.5) lets admission scale chunk granularity to maintain `t_PIM(decode-attn) ≈ sum of GPU stages` across any ctx within TBT SLO, so the same Init/T1–T5 dispatch ordering recovers regardless of chunk size. The trace ceases to apply only at very long ctx where balance is infeasible (§6.6 "A-bound natural transition") — a system-level property of multi-request schedulers under growing ctx, not a PULS-specific limitation.

### 6.6 Bound Analysis

Qualitative estimation. Since the scheduler recognizes the bound at runtime via the §6.4 idle fraction, this table is not a control input. After sim measurement, only the component times · transition ctx are updated.

**Intra-A bound** — From the perspective of double-buffering (§5.6): Instance A's internal GPU stage (projection + AR) vs PIM stage (decode attention).

- **Short-ctx (2k–8k): GPU-bound.** Sum of GPU stages > t_PIM. §6.4 Intra-A prescription: admit additional decode → fill the next μ-batch's PIM window. Since AR cost is borne identically by the baseline, only the position of the transition ctx is preserved.
- **Long-ctx (≥ 32k): PIM-bound.** t_PIM > sum of GPU stages. §6.4 Intra-A prescription: admit additional prefill chunk → fill the next μ-batch's GPU stage.

**Inter-AB bound** — From the perspective of the inter-instance pipeline (§3.4): Instance A pipeline cycle vs Instance B FFN cycle. A_cycle = max(t_proj + t_AR, t_PIM) vs B_cycle = t_FFN_wide.

- **Mid-ctx (32k–256k): B-bound.** B_cycle > A_cycle. PIM attention is still shorter than the FFN cycle.
- **Long-ctx (≥ 256k): natural transition to A-bound.** A_cycle ≥ B_cycle. PIM attention surpasses the FFN cycle.

**Tile-level (per-tile, ns scale)** — The tile time of PIM decode-attn = max(FP8 load time, FSM compute time). Under FP8 KV assumption it is in the compute-bound regime; under FP16 KV assumption it transitions to the load-bound regime, and the tile time roughly 2×'s. FP8 quantization is the direct enabler of this t_decode-attn_PIM.

### 6.7 Implementation Requirements

- On top of an open-source LLM serving simulator (Vidur) fork: 1 event queue, 1 dependency DAG, 3-μ-batch state in the in-flight window. Same invocation cadence as the production scheduler step.
- PIM completion-time predictor (FSM cycle-accurate).
- Idle fraction telemetry (per GPU·PIM, accumulated per iteration).
- Admission controller (dynamic adjustment of chunk size · decode batch).

## 7. Orthogonality to Complementary Techniques

### 7.1 Paged KV Memory Management

- **Layer distinction:** Paged KV management is a *memory-management* layer that manages the KV cache as non-contiguous memory pages. PULS PIM is a *compute-offload* layer that executes attention operations over the KV data resident in HBM.
- **Non-interference:** The two operate at different abstraction levels and share no interface. Since the PIM FSM can transparently handle a non-contiguous KV layout via page-table reference, page-based physical placement decisions do not affect the correctness of the PIM operation.
- **Cumulative gains:** The fragmentation loss eliminated by page management and the GPU-side attention cost eliminated by PIM accumulate independently.

### 7.2 Speculative Attention

- **Technique definition:** Speculative decoding generates draft tokens and then performs parallel verification in a single forward pass. Speculative attention optimizes the attention cost of this verification pass.
- **PIM applicability:** Since PULS PIM processes the attention operation uniformly regardless of token origin (draft / verified / position within the speculation tree), PIM offload holds for the speculative attention pass as well.
- **Combined gains:** Speculative decoding reduces the number of forward passes, and PIM lowers the attention cost of each pass. The throughput improvement from combining the two optimizations works multiplicatively.

### 7.3 Prefix KV Caching

- **Technique definition:** A class of techniques that skip the prefill operation itself via shared-prefix KV cache hits.
- **Hit region:** The PIM's prefill-side attention load also decreases proportionally.
- **Miss and decode regions:** The PIM offload gain is preserved as is.
- **Cumulative gains:** Prefix caching shrinks the overall compute scale via KV reuse, and PULS absorbs the attention cost of the remaining computation. The KV hit rate and the PIM offload gain are independent variables that contribute to performance with no cross term.

## 8. Open Empirical Work

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

All quantitative figures in this architecture document (acceleration multiples, latency / throughput absolute values, MFU plateau, admission ceiling values, deadband width %) **will be measured in the Phase 3 calibration track**.
