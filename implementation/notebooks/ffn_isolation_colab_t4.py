# ============================================================================
# FFN-only Colab T4 Microbench — Aux1 mixed batching weight reuse ratio
# ============================================================================
#
# 목적 — PULS RFC Impl-10 main (Stage 2) D2 deliverable #5:
#   Llama-3 70B 단일 layer FFN 위 *mixed batch 1 GEMM* vs *separate 2 GEMM*
#   영역의 architectural property (weight streaming 절감) 의 empirical anchor.
#
# 환경 — Google Colab Free Tier (T4 GPU 16 GB).
#   Runtime → Change runtime type → Hardware accelerator: T4 GPU.
#
# 산출 — `aux1_ratio_t4_measured.json`:
#   - batch_sweep_ms: per-batch 단일 layer FFN forward time
#   - aux1_ratio: t_separate / t_mixed (mixed batching 가속 ratio)
#
# 출처 라벨 — `colab_t4_free_llama3_70b_ffn_single_layer_measured`.
#
# 영역 정합 — Stage 2 markdown 위 *empirical anchor disclosure* 만, 코드 산식
# 영역 ingest 0 (T4 ≠ B200 영역 mismatch + PLAN §0.5 ratio property 정합).
#
# ============================================================================

import json
import time
from datetime import datetime

import torch

# ---------------------------------------------------------------------------
# 0. 환경 영역 검증 (T4 + CUDA 가용성)
# ---------------------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA 미가용 — Runtime 영역 GPU (T4) 활성 필요"
DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
print(f"[env] device = {DEVICE} ({GPU_NAME})")
print(f"[env] torch = {torch.__version__}, CUDA = {torch.version.cuda}")
print(f"[env] memory total = {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---------------------------------------------------------------------------
# 1. Llama-3 70B FFN spec (ARCH §3.4 + Llama-3 published spec)
# ---------------------------------------------------------------------------

HIDDEN = 8192            # Llama-3 70B hidden_size
INTERMEDIATE = 28672     # Llama-3 70B FFN intermediate_size (SwiGLU)
DTYPE = torch.float16    # FP16 (T4 위 tensor core 정합)

# 단일 layer FFN 3 GEMM weights — gate + up + down
# Memory 영역 산출:
#   gate: 8192 × 28672 × 2 B ≈ 470 MB
#   up:   8192 × 28672 × 2 B ≈ 470 MB
#   down: 28672 × 8192 × 2 B ≈ 470 MB
#   total ≈ 1.4 GB FP16 (T4 16 GB 위 여유)
print(f"\n[spec] Llama-3 70B FFN: hidden={HIDDEN}, intermediate={INTERMEDIATE}")
print(f"[spec] dtype={DTYPE}, weights ≈ 1.4 GB (3 GEMM × ~470 MB each)")

torch.manual_seed(42)
gate_w = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device=DEVICE) * 0.02
up_w = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device=DEVICE) * 0.02
down_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.02

mem_after_weights = torch.cuda.memory_allocated() / 1e9
print(f"[spec] memory after weights = {mem_after_weights:.2f} GB")


# ---------------------------------------------------------------------------
# 2. SwiGLU FFN forward — 3 GEMM
# ---------------------------------------------------------------------------

def ffn_one_pass(x: torch.Tensor) -> torch.Tensor:
    """Llama-3 SwiGLU FFN forward — 1 layer.

    y = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    """
    gate = x @ gate_w                              # [B, intermediate]
    up = x @ up_w                                  # [B, intermediate]
    hidden = torch.nn.functional.silu(gate) * up   # [B, intermediate]
    out = hidden @ down_w                          # [B, hidden]
    return out


# ---------------------------------------------------------------------------
# 3. CUDA Event timing — warmup + median 영역
# ---------------------------------------------------------------------------

def measure_ms(fn, n_warmup: int = 10, n_iter: int = 100) -> float:
    """CUDA Event 위 정밀 timing. Median 영역의 *mean 위 단순화* 영역 (안정 영역 확보)."""
    # warmup
    for _ in range(n_warmup):
        _ = fn()
    torch.cuda.synchronize()

    # measure (single block — n_iter 영역 위 mean)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter  # ms per iter


# ---------------------------------------------------------------------------
# 4. Batch sweep — T4 N_sat + B200 N_sat 양 영역 cover
# ---------------------------------------------------------------------------

print("\n[sweep 1] batch sweep — single layer FFN forward time")
batch_sweep = [1, 16, 64, 256, 1024]
batch_sweep_results = {}

for batch in batch_sweep:
    try:
        x = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.02
        ms = measure_ms(lambda: ffn_one_pass(x))
        batch_sweep_results[batch] = {
            "t_ms": ms,
            "memory_alloc_GB": torch.cuda.memory_allocated() / 1e9,
        }
        print(f"  batch={batch:5d} → {ms:8.4f} ms  (mem={torch.cuda.memory_allocated()/1e9:.2f} GB)")
        del x
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  batch={batch:5d} → OOM (T4 영역 한도 초과 — 본 batch 제외)")
        batch_sweep_results[batch] = {"t_ms": None, "memory_alloc_GB": None, "oom": True}


# ---------------------------------------------------------------------------
# 5. Aux1 ratio sweep — mixed batch (2B) vs separate batch (B × 2)
# ---------------------------------------------------------------------------

print("\n[sweep 2] Aux1 ratio — t_separate / t_mixed (mixed batching architectural property)")
mixed_vs_separate = [(8, 16), (32, 64), (128, 256), (512, 1024)]
aux1_ratio_results = {}

for k, two_k in mixed_vs_separate:
    try:
        x_k = torch.randn(k, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.02
        x_2k = torch.randn(two_k, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.02

        t_k_ms = measure_ms(lambda: ffn_one_pass(x_k))
        t_2k_ms = measure_ms(lambda: ffn_one_pass(x_2k))
        t_separate_ms = 2 * t_k_ms  # separate 2 GEMM 영역
        t_mixed_ms = t_2k_ms        # mixed batch 1 GEMM
        ratio = t_separate_ms / t_mixed_ms

        aux1_ratio_results[f"{k}_to_{two_k}"] = {
            "k": k,
            "two_k": two_k,
            "t_k_ms": t_k_ms,
            "t_2k_ms": t_2k_ms,
            "t_separate_ms": t_separate_ms,
            "t_mixed_ms": t_mixed_ms,
            "ratio_separate_over_mixed": ratio,
        }
        print(
            f"  k={k:5d} → 2k={two_k:5d}  "
            f"t_sep={t_separate_ms:8.4f} ms, t_mix={t_mixed_ms:8.4f} ms, ratio={ratio:.3f}"
        )
        del x_k, x_2k
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  k={k} → 2k={two_k}: OOM")
        aux1_ratio_results[f"{k}_to_{two_k}"] = {"oom": True}


# ---------------------------------------------------------------------------
# 6. JSON 산출 + Colab download
# ---------------------------------------------------------------------------

output = {
    "metadata": {
        "provenance_label": "colab_t4_free_llama3_70b_ffn_single_layer_measured",
        "gpu_name": GPU_NAME,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_spec": {
            "name": "Llama-3 70B FFN single layer",
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "dtype": "float16",
        },
        "measurement_method": {
            "timer": "torch.cuda.Event(enable_timing=True)",
            "n_warmup": 10,
            "n_iter": 100,
            "aggregation": "mean over n_iter (single-block CUDA Event)",
        },
        "disclosure": {
            "scope": "Stage 2 D2 deliverable #5 — Aux1 empirical anchor",
            "ingest_policy": "report markdown 위 disclosure 만 — 코드 산식 영역 ingest 0",
            "rationale": "T4 ≠ B200 substrate mismatch + PLAN §0.5 ratio property 정합",
        },
    },
    "batch_sweep_ms": batch_sweep_results,
    "aux1_ratio": aux1_ratio_results,
}

OUTPUT_PATH = "aux1_ratio_t4_measured.json"
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[done] saved → {OUTPUT_PATH}")
print("\n=== JSON preview ===")
print(json.dumps(output, indent=2)[:2000])
print("...\n")


# ---------------------------------------------------------------------------
# 7. Colab download (마지막 cell 위 실행 — 자동 download trigger)
# ---------------------------------------------------------------------------
#
# Colab 환경 위:
#     from google.colab import files
#     files.download(OUTPUT_PATH)
#
# 또는 manual — Colab 좌측 영역 *Files* 탭 위 `aux1_ratio_t4_measured.json` 우클릭 → Download.
#
# 다운로드 후 본 puls-rfc workspace 위 저장 위치:
#     implementation/data/aux1_ratio_t4_measured.json

try:
    from google.colab import files  # type: ignore
    files.download(OUTPUT_PATH)
    print("[colab] download triggered")
except ImportError:
    print("[colab] non-Colab 환경 — manual download 영역 (위 instructions 참조)")
