# PULS — PIM-Unified LLM Serving

**Scheduler-aware co-design of HBM-PIM and production LLM serving stack.**

> **Disclosure** — Personal research project by a single undergraduate author. No institutional / vendor affiliation.
>
> Self-study work in progress. Feedback, critique, and mentorship are warmly welcomed (GitHub Issues / Discussions).

📊 **Interactive visual guide:** [**rhydcdc.github.io/puls-rfc/scheduler_policy.html**](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html)

This repo is the public RFC (Request for Comments) of an in-progress prototype. It provides the entry point of the architecture + design rationale + scheduler policy, together with a calibrated projection of the four acceleration sources on real long-context traces (see [Results](#results)).

Full body — [`ARCHITECTURE.md`](ARCHITECTURE.md) (substrate, instance disaggregation, scheduler integration, adaptive admission, layer flow, prior art comparison, all integrated).

> **Update log.**
> - **Upgrade 1 — scheduler logic generalized.** Model/HW-variabilized `derive` + a `runtime` / `sim` / `validation` split under a single CONTRACT ([`puls-engine`](puls-engine/CONTRACT.md), 197 checks). The operating point is derived, not hand-set.
> - **Upgrade 2 — global scheduler + instance-dependency model.** Added a **global age-cap** (cross-node FIFO fairness → forced routing) and an **on-node multi-turn KV cache**, and modeled the **intra-/inter-instance dependency** in TBT (Instance A→B dependency + PIM↔GPU-A HBM contention). The winning cluster design is **C**; its config and numbers are in [Cluster Scheduler (C)](#cluster-scheduler-c).
> - **Upgrade 3 — physical cache affinity + honest accounting.** Multi-turn returns are now routed to the node that physically holds their KV in HBM (affinity queue, 1st priority on hole; separate spill cap 200 vs global age-cap 25 — waiting has a reward for cached returns), the cache budget is dynamically debited by multi-turn pool-KV inflation (~0.24 TB measured), the SSD reload constant is corrected to a realistic NVMe-array value (2e7 → 1e8 B/round; the old value sat 5% below the recompute break-even 2.1e7, making tier 2 mathematically useless), and a per-request token-gap KPI ([PAUSE]) closes the pause-cost blind spot of batch-level TBT. Physical cache hits 1.7% → 99.7%. Numbers in [Cluster Scheduler (C)](#cluster-scheduler-c).
> - **Upgrade 4 — 2-active prefill consistency + prefill_pool re-derivation.** Both active μ-batches carry 128 prefill tokens (256/round, request-disjoint — chunk k+1 needs chunk k's KV); the lifecycle sim's once-per-round prefill accounting is fixed to 2×128, and prefill_pool is re-derived 60→80 (batch-1's greedy cherry-picks near-ideal depths, starving batch-2's residual depth coverage: 60 → batch2 96.3%/1.84%; 80 = knee 99.06%/0.54%). Prefill throughput doubles (transitions ×1.9). Instance A 3.02→3.20 TB (16-high only), cache budget −0.18 TB → hbmHit 91.1→87.9%, TTFT +1.5% — the honest bill for previously unaccounted prefill. Numbers updated throughout.

## Table of Contents

**Background**
- [Characteristics of Processing-in-Memory Architecture](#characteristics-of-processing-in-memory-architecture)
- [Problem Statement](#problem-statement)

**The Proposal**
- [The PULS Proposal](#the-puls-proposal)
- [Approach Summary](#approach-summary)
- [Workload Coverage](#workload-coverage)
- [Acceleration Sources Summary](#acceleration-sources-summary)
- [Role of Mixed Batching](#role-of-mixed-batching)

**Results**
- [Results](#results)
- [Runtime Validation](#runtime-validation)
- [Cluster Scheduler (C)](#cluster-scheduler-c)
- [Limitations / Disclosure](#limitations--disclosure)

**Resources**
- [Interactive Reading Guide](#interactive-reading-guide)

## Characteristics of Processing-in-Memory Architecture

*This Section summarizes the structural constraints shared by every PIM architecture.* From HBM4E onward, custom logic-die fabrication begins. To execute MAC operations in parallel with ongoing GPU processing, PIM must place MAC units on either **the logic die (PHY) or the DRAM die of HBM**. This substrate choice imposes the following structural constraints:

- **Logic die (PHY) placement — area constraint** — Logic die area is limited; a meaningful quantity of compute units (general-purpose GEMV / GEMM scale) cannot be installed.
- **DRAM die placement — memory capacity reduction** — Adding MAC units to the DRAM die encroaches on memory cell area, reducing available KV cache capacity.
- **Thermal envelope constraint** — DRAM is heat-sensitive and cannot sustain large MAC workloads. Throttling necessarily accompanies PIM activation.
- **General-purpose computation unrealistic** — The combination of the three constraints above makes it unrealistic to absorb all general-purpose computation into PIM. Careful scoping of the PIM workload is mandatory.
- **Shared memory path — GPU ↔ PIM cannot occupy simultaneously** — Memory data cannot be simultaneously occupied by the GPU and the PIM core. Both sides must traverse the same shared path (TSV or inter-bank path), which becomes a potential source of latency in GPU-side data loads.

## Problem Statement

Limitations of existing HBM-PIM research, grouped by axis:

**Substrate-level (cell · bank · die)**
- **Memory-die intrusion** — PIM logic encroaches on the memory die, reducing available KV cache capacity.
- **Heat / thermal envelope constraint** — Thermal throttling during PIM activation halts PIM operation.
- **Non-standard DRAM circuit modifications required** — Dual row buffers, bank-level compute logic, and similar cell- / bank-level circuit changes deviate from standard DRAM fabrication, raising fab risk and yield concerns beyond ordinary logic ASIC integration.

**System-level (interconnect · resources · auxiliary HW)**
- **Complex auxiliary scheduling logic + extra hardware modules** — Bespoke controllers, DMA engines, and similar non-substrate components are required.
- **Logic die ↔ DRAM die / HBM ↔ host round-trip traffic** — Intermediate attention results (softmax accumulator, row max, etc.) shuttle between the logic die and DRAM die, and between HBM and the host (GPU / NPU), occupying the internal bus and the host ↔ HBM path.
- **Structural requirement of dedicated large-scale HBM resource beyond the GPU (in some designs)** — Designs that physically separate PIM-enabled HBM from the GPU's HBM impose an additional, dedicated HBM provision on top of the host GPU memory.

**GPU ↔ PIM concurrency (stall · path contention)**
- **GPU stall during PIM execution (in some designs)** — Some designs explicitly block the GPU while the PIM completes its work, serializing what should be parallel resources. The GPU's idle time during this window cannot be repurposed.
- **Shared memory path contention** — PIM and GPU share the same physical path (TSV / C/A bus / inter-bank path / row buffer). PIM activation contends with the GPU's concurrent HBM reads (weight streaming, KV transactions), slowing the GPU side.

**Serving lifecycle (KV cache continuity · session state)**
- **Multi-turn / continuation prefill not supported (in many existing designs)** — Existing designs commonly assume a single Sum → repeated Gen flow per request; continuation prefill that must attend to existing KV (multi-turn chat, chunked prefill, RAG with growing context) is not architecturally accommodated.
- **Bulk KV cache transfer between GPU and dedicated PIM resource (in some designs)** — When PIM-side HBM is physically separated from the GPU, the KV cache must traverse the host ↔ PIM interconnect (PCIe / NVLink / CXL), making the interconnect a bottleneck that partly nullifies the internal-bandwidth advantage of PIM.

**Workload coverage (model variants · batch regime)**
- **Incompatibility with modern serving features** — Lacks support for standard production-serving features such as GQA and speculative decoding.
- **Unstable performance advantage across batch sizes (in many existing designs)** — Across a batch-size sweep, the PIM-advantageous regime is narrow and the transition is discontinuous.
- **Inherent throughput limit of PIM attention** — Even when attention is offloaded to PIM, op-level token throughput remains limited.

## The PULS Proposal

*Why attention specifically?* Attention is universal across transformer-based models and admits online streaming compute — a natural fit for the PIM substrate.

- **HBM-PIM architecture** — substrate design that avoids the above limitations:
  - **Memory-die non-intrusion (P1)** — avoids encroachment on KV cache capacity
  - **Row-wise pipelined FSM (deterministic cycles)** — secures thermal-envelope predictability; PIM completion time is precomputed at dispatch
  - **Per-channel PIM / GPU toggle** — no separate DMA engine / bespoke controller required
  - **SP-PIM cross-GPU 2048-channel cooperation** — distributes the single-op-level throughput limit (8 GPU lock-step cooperation)
  - **GQA / speculative-attention compatibility** — modern serving features supported
  - **Row-wise accumulation inside the logic-die SFU** — avoids logic die ↔ DRAM die round-trip traffic
  - **Compute-bound timing alignment (P5)** — avoids TSV contention; PIM activation is confined to windows when the GPU is executing compute-bound ops
- **In-house scheduler** — compatible with chunked-prefill + mixed-batch primitives ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 6):
  - **Event-driven dispatch + dependency DAG** — encodes invariants (data dependency + resource) as a graph, reducing dispatch to a ready-node selection problem
  - **2-μ-batch lookahead** — starting work for the next μ-batch early + filling idle resources with work from another μ-batch (emerges naturally)
  - **Pool-model composition with local-greedy steering (node level)** — admission (pool refill) ‖ decode-set steering ‖ prefill steering, each hitting fixed operating-point targets independently, with an age-cap for fairness (no global statistics, no idle-feedback loop)
  - **Global scheduler (server-level node distribution)** — routes the full arrival pool to nodes: **greedy cold-start** (shed long requests to edge nodes + interleave-greedy to fill normal nodes to mean 100K) + **greedy healing** (per-completion like-for-like refill = toxic-fit, preserving the long-request distribution, drift 0). **Sending only the first ~2.2% to edge nodes then keeps each node's pool at the mean-100K operating point** so local steering hits — inter-node movement / eviction = 0 ([`ARCHITECTURE.md`](ARCHITECTURE.md) §7)
  - **Preemption-free deterministic admission** — full-length KV is reserved at admission, so an admitted request is *never evicted / recomputed* (zero lost work). Memory pressure is absorbed by admission backpressure (refusing new requests), not preemption, and the age-cap bounds waiting (starvation-free). The two bounds — space (KV cap) and time (age-cap) — are orthogonal: pressure on one axis is never repaid by reclaiming the other

## Approach Summary

![PULS Instance Disaggregation](figures/instance_disaggregation.png)

- **Instance disaggregation** — The transformer layer is split into Instance A (attention block, 8 GPU, TP=8 + SP-PIM 2048 channels) and Instance B (post-attention block / FFN, 8 GPU, TP=8). The two instances are connected serially via an inter-instance pipeline. The KV cache resides permanently in Instance A's HBM (no inter-instance KV transfer). Additional effects that this split yields:
  - **Instance B substrate-cost reduction potential** — Instance B is FFN compute-bound, so HBM's full bandwidth does not contribute to B_cycle. Substituting **GDDR (or low-stack HBM)** can reduce unit cost ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 3.4 "Instance B Memory Substrate").
  - **Fixed-shape tensor handoff → straggler-bubble elimination** — KV-length variance produces variable-shape tensors in the decode batch, but PIM absorbs the length dependency at the attention stage, and only fixed-shape `[B × hidden]` tensors are passed to Instance B. Instance B performs uniform GEMMs without ragged batching, eliminating decode-batch straggler bubbles ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 5.2).
  - **KV-cache-movement bus-transaction reduction** — Since PIM processes attention in place, GPU ↔ HBM KV-streaming traffic is eliminated outright. In the long-context regime, where KV bytes read is the determining factor of attention time, the effect is large. **Incidental thermal reduction expected** (bus-transaction energy ≫ compute energy) ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 5.7 auxiliary item).
  - **FFN-weight bus-transaction reduction** — When PIM absorbs the attention length dependency and mixed batching is reinstated, prefill chunks and decode tokens share the same FFN weights. The token denominator of weight HBM traffic grows, reducing per-token FFN-weight bus transactions + raising arithmetic intensity. **Incidental thermal reduction expected** ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 5.7 auxiliary item).
- **HBM4 logic-die SP-PIM**:
  - **Compute substrate** — row-wise pipelined attention SFU, 32-row tile FSM @ 1.3 GHz, FP16 MAC + FP8 (E4M3) KV cache (aligned with the production FP8 KV regime)
  - **Channel-level PIM / GPU toggle** — Each of the 32 channels per HBM4 stack can be independently toggled between PIM and normal mode
  - **SP-PIM cross-GPU cooperation (Q-replicate parallelization)** — Instance A's 8 GPU × 256 channel = 2048 channels in total cooperatively process a single attention operation in lock-step. The same Q vector is broadcast to all 2048 channels and KV rows are sharded across channels, so each channel independently sweeps its own KV slice → a single attention operation executes in parallel across 2048 channels
  - **Row-wise accumulation inside the logic-die SFU** — Intermediate attention results (softmax accumulator, row max) accumulate inside the SFU, avoiding logic die ↔ DRAM die round-trip traffic
- **Interceptor host↔PIM interface** — A direct consequence of confining PIM to the single decode-attention operation ([`ARCHITECTURE.md`](ARCHITECTURE.md) Section 3.5):
  - By **claiming a single RFU (Reserved For Future use) bit of the existing JEDEC HBM4 RD/WR commands as PIM_toggle**, the host↔PIM interface is absorbed into the standard DRAM command set. No separate interrupt required.
  - Based on the FSM's deterministic cycles, **computed wait** lets the GPU precompute when the result is to be read.
  - The only channel between PIM and GPU is HBM (PIM write → GPU read, the same pattern as inter-kernel data passing through global memory on the GPU).

Full body — [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Workload Coverage

PULS is **not confined to a specific target workload**, because batch composition is *length-distribution-agnostic* steering:

- **Steering never looks at the pool's mean length** — it combines short and long requests to hit only the four operating-point targets (deployed 128: decode count 62 ∧ Σkv 6.15M, prefill 128 tokens ∧ depth-work 12.8M; OPERATING_POINT §4.1).
- Hence **any distributed-server-scale workload with decode and prefill abundant in the pool** — short, long, mixed, or bimodal length distribution alike — converges to the operating point.
- (The balance ctx ~100K is merely the midpoint used to *derive* the KV cap, not a value imposed on the workload.)
- **Condition for reaching the operating point (idle ≈ 0)** = decode and prefill are *abundant* in the pool (the large standing decode population + continuous prefill of high-concurrency distributed serving). Real-server steady state is exactly this regime. Demonstrated by an integrated sim with **2 active μ-batch + dependency · age-cap** composing to the operating point at **decode 99.92% · Σdev 1.28% / prefill ≈99.5% · Σdev 0.32%** — see [Runtime Validation](#runtime-validation).
- **When the pool is thin** (low load, short-decode only), PIM or GPU-A idling is *physically normal* (not something to fix) — the operating point is the balance point that holds when the pool is abundant.

## Acceleration Sources Summary

| ID | Source | Scope |
|---|---|---|
| F1 | SP-PIM attention (2048-channel Q-replicate, 8 GPU lock-step) | Op-level |
| F2 | Projection ‖ PIM attention double-buffering (intra-instance) | Op-level |
| F3 | Instance A–B inter-instance pipeline (steady-state) | Systems-level |
| F4 | μ-batch staggering (steady-state precondition for F2·F3) | Systems-level |
| F5 | Channel-independent PIM scheduling (absorbs KV-length variance) | Systems-level |
| (Aux) | Mixed batching resurrection (shared weights → arithmetic intensity ↑) | TTFT / throughput trade-off |
| (Aux) | Bus traffic reduction (PIM in-place attention → HBM-GPU bus transactions ↓) | Energy / cost |

## Role of Mixed Batching

- **Primary purpose: securing the PIM TSV-occupancy window (compute-bound headroom)** — Under pure decode alone, projection switches to memory-bound, and the PIM headroom may be insufficient. Admit additional prefill-chunk tokens into the mixed batch to push the effective N into the GEMM-saturating regime, keeping projection in a compute-bound regime.
- **Side effects (derived from weight sharing)**:
  - prefill chunks and decode tokens share the same weights → arithmetic intensity rises
  - per-token FFN-weight bus transactions decrease
  - inter-instance KV transfer is eliminated (coexistence within a single instance)

## Results

Calibrated projection of the four acceleration sources (Aux1·Aux2·F3·F5) on Llama-3 70B + DGX B200 + HBM4 substrate, plus a runtime validation pass (integrated lifecycle sim on a synthetic distributed-server steady-state workload).

> **Visualized companion** — Per-source schematic figures + numerical breakdown + runtime validation + honest disclosure are organized in the interactive reading guide as **§6 Acceleration Sources** and **§7 Results**: [`docs/scheduler_policy.html`](docs/scheduler_policy.html) ([online](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html#sec-7)).

### Substrate

| Component | Value |
|---|---|
| GPU compute | DGX B200 standalone, FP16 dense 2,200 TFLOPS per GPU |
| Memory | HBM4 hypothetical projection, 16 TB/s per GPU × 8 GPU = 128 TB/s aggregate |
| η_HBM_external | 0.74 (fix) + sensitivity sweep {0.70, 0.74, 0.80} |
| PIM tile time | 267 ns (compute-bound, FP8 KV) |
| RTL FSM | 347 cycles @ 1.3 GHz |
| Model | Llama-3 70B (L=80, hidden=8192, FFN intermediate=28672) |

### Per-source Acceleration

| Source | Mechanism | Result |
|---|---|---|
| Aux1 | Mixed batching weight reuse | **2.0×** (closed-form), 1.97× (Colab T4 measured) |
| Aux2 | KV bus traffic reduction | **4.95× speedup, 79.8% reduction** |
| F3 | Inter-instance pipeline ratio | 0.92–0.99 (closed-form ctx sweep); the integrated lifecycle sim composes **2 active μ-batches to the operating point, decode 99.92% · Σdev 1.28% / prefill ≈99.5% · Σdev 0.32% (balance manifest)** — see [Runtime Validation](#runtime-validation) |
| F5 | Channel-independent vs lock-step | **5.15× speedup** (KV variance dominant) |

### Aggregate Speedup

| Component | Baseline | PULS | Saving |
|---|---|---|---|
| Weight streaming | 2.89 ms | 1.45 ms | 1.45 ms |
| KV bus traffic | 8.25 ms | 1.67 ms | 6.59 ms |
| **Total (weight + bus)** | **11.14 ms** | **3.12 ms** | **8.04 ms (72.2%)** |

Net speedup: **3.57× (closed-form, weight + bus)** → **4–5× (including F5)**.

### Runtime Validation

**Integrated lifecycle simulation (PULS-independent, composition-hit only).** Rather than a single real trace, a synthetic workload representing the distributed-server steady state is *fixed*, and cold-start → operation (steering · prefill→decode transition · per-completion healing · age-cap) is run in one sim to check whether the operating-point composition holds — the seed is not tuned to a predetermined answer.

- **Workload (synthetic)** — a wide, diverse-length pool (1K–1M, short/mid/long mixed), prefill and decode abundant. **warm-start** = a steady-state snapshot (each request placed at a random lifecycle point, skipping the cold-start ramp).
- **Model — 2 active μ-batch (not 3).**
  - A node runs only 2 μ-batches concurrently (F2/F3 overlap).
  - When one batch's forward pass ends, its members return to the pool and **a new batch is re-selected from (the returned members + the standing surplus)** (zero memory allocation) — *no third batch is force-composed.*
  - Completed requests are refilled like-for-like by per-completion healing, completed prefills transition to decode, age-cap = 5 for fairness. Deployed prefill 2×128 (both active batches, request-disjoint — chunk k+1 needs chunk k's KV).

**composition — operating-point hit ([puls-engine/sim/lifecycle.cpp](puls-engine/sim/lifecycle.cpp), with dependency · age-cap; repro: `puls_lifecycle 4000 64 5 2000`):**

| 2 active μ-batch (re-composed on completion) | operating-point target | hit | Σdev |
|---|---|---|---|
| **decode** | 62 ∧ Σkv 6.15M | **99.92%** | **1.280%** |
| **prefill b1 / b2** | 128 tokens ∧ depth-work 12.8M (2-active 2×128, request-disjoint) | **100% / 99.06%** | **0.107% / 0.537%** |

> **(2026-06 sim-faithfulness correction)** The earlier decode 100% / Σdev 0.38% came from a healing bug in the lifecycle sim (centering admit `ideal≈ctx_balance` collapsed the pool to all-mid, starving long requests to 0% → composition trivially perfect). With the canonical healing (per-completion `ideal=hole`, like-for-like) + edge gating + prompt-independent realistic decode lengths + best-of-2000 infinite-pool emulation, on a distribution-preserving (≈20/70/10) diverse pool the deployment point (count 62, age_cap 5) gives decode 99.92% / Σdev 1.28% — consistent with the §3 age-cap sweep's cap5 spread (0.7%). Prefill (2-active 2×128) combined ≈99.5% / Σdev 0.32% (b1 100%/0.107% · b2 99.06%/0.537%).

Interpretation:

- **No age-cap tail (the 2-active structure).** The old validation's "third-batch spike (count 108 · spread 3.7%)" came from a *forced 3rd batch* where warm-start waiting members triggered the age-cap. The 2-active model never forces a 3rd batch (it only *re-composes* from returned members + surplus), so that tail vanishes structurally.
- **Holds even with dependency · age-cap.** Adding the prefill→decode transition and the fairness age-cap = 5 does not break either composition — the logic (steering · greedy · healing · age-cap · KV-centering) is *scale-invariant*; only the operating-point constants change (prefill 256 ↔ 128 isomorphic).
- **Throughput is sustained by construction.**
  - The per-cycle decode budget is pinned to the operating point (62, or fewer when the KV cap binds first), so as long as the pool stays abundant each cycle processes a fixed quantum; sustainability is a structural consequence of the fixed operating point.
  - Absolute tok/s awaits silicon-calibrated cycle time.
- **Cluster scale — global scheduler.**
  - At server scale (hundreds–thousands of nodes) the global arrival mean exceeds 100K, so per-node pools drift and idle blows up.
  - The **global scheduler** (greedy cold-start: shed long to edge + interleave-greedy · per-completion healing = toxic-fit) **pays only the ~2.2% initial edge cost and thereafter holds each node at the mean-100K operating point indefinitely** (long requests preserved, drift 0).
  - Principle, E sweep, and on2 measurement in [`ARCHITECTURE.md`](ARCHITECTURE.md) §7 / [puls-engine/core/global_scheduler.cpp](puls-engine/core/global_scheduler.cpp).

## Cluster Scheduler (C)

The winning cluster design **C** keeps the §6 node mechanism (resident **surplus** + per-iteration steering re-selection + per-completion healing) and adds two cluster-level pieces: a **global age-cap** (cross-node FIFO fairness → forced routing) and an **on-node multi-turn KV cache** (3-tier HBM / SSD / recompute, placed in the HBM left after the decode pool).

- **The three split the labor** — surplus → composition (Σdev), global age-cap → latency / fairness (and it is what lets returning sessions be served fast enough to hit the cache), cache → TTFT.
- **Surplus re-selection also proved to be a *general-purpose absorber*** of arbitrary batch perturbations — age-cap forced injections and affinity length-mismatches alike — which is what let cache affinity be added with zero new machinery.
- **The same TBT now carries the instance-dependency model** — Instance A (PIM ‖ GPU-A) must finish before Instance B (FFN), and PIM↔GPU-A share HBM, so `TBT = max(instance_a, t_ffn) × layers` with `instance_a = max(t_pim, t_gpu_a) + β·max(0, t_pim − t_gpu_a)` (contention-free when `t_pim ≤ t_gpu_a`).

> **Two modeling axes — do not conflate.** `max_tokens` models the **random EOS appearance**: each request's decode-loop length on the time axis (how many tokens until EOS — the random EOS replaced by a sampled deterministic constant), *not* an amount of KV. The **KV** (`live_kv = prompt + dec`, ≈6.2M) is the amount read **per layer** every step — PIM reads it in per-layer attention, so `TBT = (per-layer max) × 80 layers`.

**Cache affinity (multi-turn returns).** Returns whose KV is HBM-cached are queued at the holding node and admitted with 1st priority when a hole appears — making **99.7%** of hits physical (the old accounting counted any-node hits; physically only 1.7% ≈ 1/Z landed on the holder).

```
hole appears at node z
  1st priority: z's affinity queue non-empty → admit the longest-waiting member (length · cap_room ignored)
  2nd priority: queue empty → queue.pull_slot(hole) as usual
          (forced if a global aged member exists, else nearest to ideal=hole)
```

Why it works:

- Holes arrive ~every 17 rounds per node, so waiting is short and naturally bounded.
- Waiting costs ~2 ms/round vs abandoning the cache costs an SSD reload of 160–780 rounds (~10× more) — so a separate spill cap (200, vs global 25) lets cached returns wait. Measured spill is near zero: **4 of 13,483 affinity candidates (0.03%)** — under the single global cap 25 it was 31% (4,317); the dedicated-cap separation cut it ~1,000×.
- Affinity returns are near-like-for-like by construction (former residents of that node, growth ≤12K, and eligibility ≥16K filters out high-growth shorts), so the distribution is almost preserved.
- The residual mismatch is absorbed by the existing surplus + re-selection + node age-cap machinery — Σdev actually improves vs spilling (spill → global forced injection is a worse perturbation).

Measured at the deployed operating point ([puls-engine/sim/csched.cpp](puls-engine/sim/csched.cpp), 8000 iters · Z = 64 · `csched 8000 64 16000 200 25 300 0.5 1e8 5 25`):

| config (locked) | value | | KPI (C, measured) | value |
|---|---|---|---|---|
| decode_surplus | **25** (decode_pool 149) | | Σdev (avg / worst) | **1.42% / 22.6%** |
| global_age_cap | **25** | | TBT (mean / p99) | **2059 / 2223 µs** |
| aff_spill | **200** | | TTFT (mean / p99) | **0.78M / 8.96M µs** |
| affinity / dyncache | **on** | | SLO goodput | **3.79M tok/s** |
| eligibility | **16000** (mid · long) | | TTFT-met | **97.0%** |
| evict_age | 200 | | cache HBM-hit | **87.9% (physical 99.7%)** |
| contention β | 0.5 (conservative; C is β-robust) | | max wait | **26 rounds** |
| offload_bw | 1e8 (SSD ≈ 50 GB/s NVMe array) | | worst token gap | **6 rounds** (= node age-cap+1) |
| node_age_cap | 5 | | | |

- **[PAUSE] KPI — per-request token gap.** Invisible to batch-level TBT (TBT p99 identical across the node-cap sweep); the node age-cap bounds the worst gap at exactly cap+1 rounds (cap 5 → 12.2 ms, 3.4% of tokens paused, mean gap 1.12) — a pure tail-bound knob.

- **Cache budget 1.075 → 0.891 TB** — prefill_pool 80 (2-active 2×128) lifts Instance A's KV reservation 3.02 → 3.20 TB (no longer fits 12-high 3.072 TB, deployed 16-high only); measured bill vs pool 60: hbmHit −3.2%p, TTFT +1.5%, goodput −0.3%.

- **C beats both alternatives on every real KPI** — the node-local baseline A (no global age-cap → TTFT 0.94M · `max_wait` 4551, unbounded starvation, length-biased; cache off) and pure pre-positioning B (no surplus → Σdev 20.1%, PIM exposed 82.5%, goodput 2.15M tok/s).
- **With the realistic SSD constant A recovers substantially** (TTFT 1.75M → 0.94M), so C's remaining edge rests on bounded waiting (26 vs 4551), composition, and the physically-real cache tier — smaller but more defensible.
- The **derived operating point is unchanged** (62 · 6.15M · ~100K · prefill 128) — surplus / age-cap / cache / β are cluster-layer knobs swept *on top of* it, not changes to it.
- Full A/B/C comparison, sweeps, and the contention analysis: [`ARCHITECTURE.md`](ARCHITECTURE.md) §7.6.

## Limitations / Disclosure

- **No hardware in hand** — No actual H100 / HBM4 silicon. Relative source decomposition is calibrated (see [Results](#results)).
- **HBM4 substrate** — hypothetical projection (current production absent; ARCH §3.1 literal alignment). JEDEC JESD270-4A spec + in-house Ramulator2 cycle-accurate measurements (FP8 / FP16 tile load · PIM compute) as references.
- **η_HBM_external** — H100 HBM3 measurement extended to HBM4 (Framing A).
- **RTL substrate** — Limited to an open-source flow (Yosys + ASAP7 + OpenSTA pre-CTS). Out of scope of commercial signoff.
- **Runtime Validation = synthetic workload** — a synthetic distribution representing the distributed-server steady state (large standing decode pool + continuous prefill) + an unbiased warm-start seed (skips the cold-start ramp). The validation target is composition (per-cycle balance).
- **Comparative baseline (vLLM / Sarathi-Serve) + F1·F2 ablation** — deferred to subsequent calibration.
- **Single-vendor production trace** — Publicly available long-context agentic production traces are effectively limited to one; augmented with a 1M-class benchmark dataset + a mid-context production chat trace as supplementary axes.
- **Main claim quantitatives = projection** — Pre-silicon numbers are *estimates* with provenance labels (see [Results](#results)).

## Interactive Reading Guide

The core scheduler policy of PULS ([`ARCHITECTURE.md`](ARCHITECTURE.md) §6) is also available as an interactive single-page companion. The guide is organized into five navigable sections — instance disaggregation, invariants & dispatch DAG, scheduler usage (pseudo-code), adaptive admission, and a worked dispatch trace example — designed to be read sequentially or jumped into by topic.

- **Online (rendered HTML, hosted via GitHub Pages):** [rhydcdc.github.io/puls-rfc/scheduler_policy.html](https://rhydcdc.github.io/puls-rfc/scheduler_policy.html)
- **Local:** clone the repo and open [`docs/scheduler_policy.html`](docs/scheduler_policy.html) in a browser

## Repository

- [`README.md`](README.md) — this document (entry point)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architecture body (motivation, design principles, substrate, instance disaggregation, scheduler integration, pool-model admission + 2-active composition validation §6.8, layer flow, prior-art comparison)
- [`OPERATING_POINT.md`](OPERATING_POINT.md) — canonical Phase-2 operating-point & batch-composition spec (pool model, steering targets, operating-point basis)
- [`puls-engine/`](puls-engine/CONTRACT.md) — model/HW-generalized C++ scheduler (derive · steering · node/global scheduler · lifecycle sim, 197 checks)
- [`docs/scheduler_policy.html`](docs/scheduler_policy.html) — interactive reading guide (companion to ARCHITECTURE.md §6)
- [`LICENSE`](LICENSE) — Apache 2.0

Build & validate: `cd puls-engine && bash build.sh` → 197 checks.

## License

Apache License 2.0. Details — [`LICENSE`](LICENSE).
