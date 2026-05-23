# PULS — PIM-Unified LLM Serving

**Scheduler-aware co-design of HBM-PIM and production LLM serving stack.**

> **Disclosure** — Personal research project by a single undergraduate author. No institutional / vendor affiliation.
>
> Self-study work in progress. Feedback, critique, and mentorship are warmly welcomed (GitHub Issues / Discussions).

This repo is the public RFC (Request for Comments) of an in-progress prototype. It provides only the entry point of the architecture + design rationale + scheduler policy; quantitative evaluation will be measured in the simulator-based simulation track.

Full body — [`ARCHITECTURE.md`](ARCHITECTURE.md) (substrate, instance disaggregation, scheduler integration, adaptive admission, layer flow, prior art comparison, all integrated).

> **Note:** `ARCHITECTURE.md` will be added in a follow-up commit (English translation in progress).

## Table of Contents

**Background**
- [Characteristics of Processing-in-Memory Architecture](#characteristics-of-processing-in-memory-architecture)
- [Problem Statement](#problem-statement)

**The Proposal**
- [The PULS Proposal](#the-puls-proposal)
- [Approach Summary](#approach-summary)
- [Target Workload](#target-workload)
- [Acceleration Sources Summary](#acceleration-sources-summary)
- [Role of Mixed Batching](#role-of-mixed-batching)

**Project Status**
- [Current Status](#current-status-2026-05-22)
- [Limitations / Disclosure](#limitations--disclosure)
- [Forward-looking: HBF-class Disaggregated Substrate](#forward-looking-hbf-class-disaggregated-substrate)

## Characteristics of Processing-in-Memory Architecture

*This Section summarizes the structural constraints shared by every PIM architecture.* From HBM4E onward, custom logic-die fabrication begins. To execute MAC operations in parallel with ongoing GPU processing, PIM must place MAC units on either **the logic die (PHY) or the DRAM die of HBM**. This substrate choice imposes the following structural constraints:

- **Logic die (PHY) placement — area constraint** — Logic die area is limited; a meaningful quantity of compute units (general-purpose GEMV / GEMM scale) cannot be installed.
- **DRAM die placement — memory capacity reduction** — Adding MAC units to the DRAM die encroaches on memory cell area, reducing available KV cache capacity.
- **Thermal envelope constraint** — DRAM is heat-sensitive and cannot sustain large MAC workloads. Throttling necessarily accompanies PIM activation.
- **General-purpose computation unrealistic** — The combination of the three constraints above makes it unrealistic to absorb all general-purpose computation into PIM. Careful scoping of the PIM workload is mandatory.
- **Shared memory path — GPU ↔ PIM cannot occupy simultaneously** — Memory data cannot be simultaneously occupied by the GPU and the PIM core. Both sides must traverse the same shared path (TSV or inter-bank path), which becomes a potential source of latency in GPU-side data loads.

## Problem Statement

Limitations of existing HBM-PIM research:

- **Limited to acceleration of a few operator kernels** — Stays at kernel-level acceleration of operators such as attention; does not translate into throughput / SLO improvement at the serving-system level.
- **Memory-die intrusion** — PIM logic encroaches on the memory die, reducing available KV cache capacity.
- **Heat / thermal envelope constraint** — Thermal throttling during PIM activation halts PIM operation.
- **Complex auxiliary scheduling logic + extra hardware modules** — Bespoke controllers, DMA engines, and similar non-substrate components are required.
- **Incompatibility with modern serving features** — Lacks support for standard production-serving features such as GQA and speculative decoding.
- **Unstable performance advantage across batch sizes** — Across a batch-size sweep, the PIM-advantageous regime is narrow and the transition is discontinuous.
- **Inherent throughput limit of PIM attention** — Even when attention is offloaded to PIM, op-level token throughput remains limited.
- **Logic die ↔ DRAM die round-trip traffic** — Intermediate attention results (softmax accumulator, row max, etc.) shuttle between the logic die and DRAM die, occupying the internal bus.

## The PULS Proposal

- **HBM-PIM architecture** — substrate design that avoids the above limitations:
  - **Memory-die non-intrusion (P1)** — avoids encroachment on KV cache capacity
  - **Row-wise pipelined FSM (deterministic cycles)** — secures thermal-envelope predictability; PIM completion time is precomputed at dispatch
  - **Per-channel PIM / GPU toggle** — no separate DMA engine / bespoke controller required
  - **SP-PIM cross-GPU 2048-channel cooperation** — distributes the single-op-level throughput limit (8 GPU lock-step cooperation)
  - **GQA / speculative-attention compatibility** — modern serving features supported
  - **Row-wise accumulation inside the logic-die SFU** — avoids logic die ↔ DRAM die round-trip traffic
  - **Compute-bound timing alignment (P5)** — avoids TSV contention; PIM activation is confined to windows when the GPU is executing compute-bound ops
- **In-house scheduler** — compatible with chunked-prefill + mixed-batch primitives ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 8):
  - **Event-driven dispatch + dependency DAG** — encodes invariants (data dependency + resource) as a graph, reducing dispatch to a ready-node selection problem
  - **2-μ-batch lookahead** — starting work for the next μ-batch early + filling idle resources with work from another μ-batch (emerges naturally)
  - **Adaptive admission with hysteresis deadband** — dynamic admission based on measured GPU / PIM idle fractions, stabilizing the PIM advantage across batch sizes

## Approach Summary

![PULS Instance Disaggregation](figures/instance_disaggregation.png)

- **Instance disaggregation** — The transformer layer is split into Instance A (attention block, 8 GPU, TP=8 + SP-PIM 2048 channels) and Instance B (post-attention block / FFN, 8 GPU, TP=8). The two instances are connected serially via an inter-instance pipeline. The KV cache resides permanently in Instance A's HBM (no inter-instance KV transfer). Additional effects that this split yields:
  - **Instance B substrate-cost reduction potential** — Instance B is FFN compute-bound, so HBM's full bandwidth does not contribute to B_cycle. Substituting **GDDR (or low-stack HBM)** can reduce unit cost ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 4.4 "Instance B Memory Substrate").
  - **Fixed-shape tensor handoff → straggler-bubble elimination** — KV-length variance produces variable-shape tensors in the decode batch, but PIM absorbs the length dependency at the attention stage, and only fixed-shape `[B × hidden]` tensors are passed to Instance B. Instance B performs uniform GEMMs without ragged batching, eliminating decode-batch straggler bubbles ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 6.3).
  - **KV-cache-movement bus-transaction reduction** — Since PIM processes attention in place, GPU ↔ HBM KV-streaming traffic is eliminated outright. In the long-context regime, where KV bytes read is the determining factor of attention time, the effect is large. **Incidental thermal reduction expected** (bus-transaction energy ≫ compute energy) ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 6.8 auxiliary item).
  - **FFN-weight bus-transaction reduction** — When PIM absorbs the attention length dependency and mixed batching is reinstated, prefill chunks and decode tokens share the same FFN weights. The token denominator of weight HBM traffic grows, reducing per-token FFN-weight bus transactions + raising arithmetic intensity. **Incidental thermal reduction expected** ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 7).
- **HBM4 logic-die SP-PIM**:
  - **Compute substrate** — row-wise pipelined attention SFU, 32-row tile FSM @ 1.3 GHz, FP16 MAC + FP8 (E4M3) KV cache (aligned with the production FP8 KV regime)
  - **Channel-level PIM / GPU toggle** — Each of the 32 channels per HBM4 stack can be independently toggled between PIM and normal mode
  - **SP-PIM cross-GPU cooperation (Q-replicate parallelization)** — Instance A's 8 GPU × 256 channel = 2048 channels in total cooperatively process a single attention operation in lock-step. The same Q vector is broadcast to all 2048 channels and KV rows are sharded across channels, so each channel independently sweeps its own KV slice → a single attention operation executes in parallel across 2048 channels
  - **Row-wise accumulation inside the logic-die SFU** — Intermediate attention results (softmax accumulator, row max) accumulate inside the SFU, avoiding logic die ↔ DRAM die round-trip traffic
- **Interceptor host↔PIM interface** — A direct consequence of confining PIM to the single decode-attention operation. By **claiming a single RFU (Reserved For Future use) bit of the existing JEDEC HBM4 RD/WR commands as PIM_toggle**, the host↔PIM interface is absorbed into the standard DRAM command set. No separate interrupt required. Based on the FSM's deterministic cycles, **computed wait** lets the GPU precompute when the result is to be read. The only channel between PIM and GPU is HBM (PIM write → GPU read, the same pattern as inter-kernel data passing through global memory on the GPU) ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 4.5).

Full body — [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Target Workload

The primary target of this RFC is the **long-context + large-batch production serving** regime — multi-turn agentic conversation, 1M-class long-context inference, high-throughput chunked-prefill + mixed-batching scenarios. In this regime, PIM's KV-cache absorption + systems-level F3·F5 effects (inter-instance pipeline, channel-independent scheduling) manifest meaningfully vs. baseline.

- **Favorable regime** — long context (≥ 32k), high batch (B ≥ 128), production traces with large KV-length variance (agentic workflows, multi-turn chat, etc.).

## Acceleration Sources Summary

| ID | Source | Scope |
|---|---|---|
| F1 | SP-PIM attention (2048-channel Q-replicate, 8 GPU lock-step) | Op-level |
| F2 | Projection ‖ PIM attention double-buffering (intra-instance) | Op-level |
| F3 | Instance A–B inter-instance pipeline (steady-state) | Systems-level |
| F4 | μ-batch staggering (steady-state precondition for F2·F3) | Systems-level |
| F5 | Channel-independent PIM scheduling (absorbs KV-length variance) | Systems-level |

F1·F2 = op-level (the regime accessible to existing op-level PIM research). **F3·F5 = systems-level (cannot be realized or measured without serving-scheduler integration)** — the core of PULS's systems-level contribution. F4 (μ-batch staggering) is not an independent acceleration source but an *enabling condition* for the steady state of F2·F3.

## Role of Mixed Batching

- **Primary purpose: securing the PIM TSV-occupancy window (compute-bound headroom)** — Under pure decode alone, projection switches to memory-bound, and the PIM headroom may be insufficient. Admit additional prefill-chunk tokens into the mixed batch to push the effective N into the GEMM-saturating regime, keeping projection in a compute-bound regime.
- **Side effects (derived from weight sharing)**:
  - prefill chunks and decode tokens share the same weights → arithmetic intensity rises
  - per-token FFN-weight bus transactions decrease
  - inter-instance KV transfer is eliminated (coexistence within a single instance)

Full details — [`ARCHITECTURE.md`](ARCHITECTURE.md) Section 7.

## Current Status (2026-05-22)

| Phase | Scope | Status |
|---|---|---|
| Phase 0 — Discovery | η_HBM, NVLink measurement, ctx-extrapolation equation, KV variance, FFN saturation, ramulator tile time, **RTL substrate design completed using the FlashAttention algorithm (online softmax + row-wise streaming) (Yosys + ASAP7 + OpenSTA pre-CTS flow)** | ✓ Closed |
| Phase 1 — Simulator Extension | PIM dispatch + FP8 KV alignment + Instance A/B scheduler + PB3 correction on top of an open-source LLM serving simulator (Vidur) fork | In progress |
| Phase 2 — Time Model and Workload | Precise PIM time model, trace replay, handoff refinement | Pending |
| Phase 3 — Calibration and Sensitivity | Sensitivity sweep, finalize quantitative figures | Pending |

The quantitative figures of this RFC (acceleration multiples, throughput / latency absolute values) **will be measured in the Phase 3 calibration track**.

## Limitations / Disclosure

- **No hardware in hand** — No actual H100 / HBM4 silicon. Evaluation runs on top of an open-source LLM serving simulator (Vidur).
- **HBM4 estimation** — Based on the JEDEC JESD270-4A spec + in-house Ramulator2-based cycle-accurate measurements (FP8 tile load / FP16 tile load / PIM compute regimes) as references.
- **RTL substrate** — Limited to an open-source flow (Yosys + ASAP7 + OpenSTA pre-CTS). Out of scope of commercial signoff.
- **Single-vendor production trace** — Publicly available long-context agentic production traces are effectively limited to one. Disclosed as a limitation; augmented with a 1M-class benchmark dataset + a mid-context production chat trace as supplementary axes.
- **Main claim quantitatives = projection** — Pre-Phase-3 numbers are *estimates*.
- **Workload-segmented deployment** — In the short-context + low-batch + pure-decode regime, projection switches to memory-bound and the PIM TSV-occupancy headroom may be insufficient. A **separated-server configuration is kept in mind**, in which short contexts (the chatbot regime) use the existing GPU-only serving stack and long contexts + large batches (agentic conversation, multi-turn) use PULS. The quantitative boundary between regimes will be decided after Phase 3 calibration.

## Forward-looking: HBF-class Disaggregated Substrate

Beyond the HBM4 SP-PIM main claim of this RFC, mounting PIM cores on a **separate memory tier such as HBF (High Bandwidth Flash)** is a direction worth further examination. Structural effects:

- **PIM duty-cycle ceiling lift** — PIM can run independently of the GPU's HBM occupancy phase, so the *compute-bound timing alignment* constraint (P5, the requirement that PIM activate only inside compute-bound windows to avoid TSV contention) is dissolved at the substrate-topology level.
- **Memory path sharing resolved** — The GPU ↔ PIM contention on the shared TSV / inter-bank path is structurally avoided through substrate-level separation: GPU keeps its HBM bus, PIM operates on the HBF bus.

That said, **HBF specifications remain unreleased at this time**, so data load latency, hot/cold KV cache tier partitioning policy, and write endurance constraints cannot be quantitatively specified. This item is *directional only*; the quantitative follow-up belongs to the period after spec disclosure.

## Repository

- [`README.md`](README.md) — this document (entry point)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architecture body (motivation, design principles, substrate, instance disaggregation, scheduler integration, adaptive admission, layer flow, prior-art comparison)
- [`LICENSE`](LICENSE) — Apache 2.0

## License

Apache License 2.0. Details — [`LICENSE`](LICENSE).
