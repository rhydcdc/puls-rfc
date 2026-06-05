// PULS-ENGINE — 고정 substrate 상수 + 변수 모델/HW 스펙.
// CONTRACT.md §1 (고정/변수 경계) · §6 (수식) 의 잠금 지점.
#pragma once
#include <algorithm>

namespace puls {

// ─────────────────────────────────────────────────────────────────────────────
// 고정 substrate — PULS 를 정의하는 상수. 모델/GPU 가 바뀌어도 불변.
// (SP-PIM + HBM4 + FP8 KV.) 출처: config.py:114-131,300-304 / OPERATING_POINT §4.1.
// 변수가 절대 이 네임스페이스로 새지 않는다(CONTRACT §9-1).
// ─────────────────────────────────────────────────────────────────────────────
namespace substrate {
    // SP-PIM (D5 RTL closure)
    constexpr double PIM_TILE_TIME_FP8_NS = 267.0;  // config.py:123 (FP8 compute-bound)
    constexpr int    PIM_TILE_ROWS        = 32;     // config.py:303 (32-row tile FSM)
    constexpr double PIM_BROADCAST_NS     = 0.5;    // config.py:304 (cross-GPU lock-step)
    constexpr int    PIM_CHANNELS_PER_STACK = 32;   // HWConfig num_channels_per_stack
    constexpr int    HBM4_STACKS_PER_GPU  = 8;      // config.py:116

    // HBM4 메모리 — 용량은 JEDEC JESD270-4A primitive 에서 *산출*(하드코딩 아님):
    //   per-stack 용량 = PIM_CHANNELS_PER_STACK(32) × channel_density(Gb) / 8  [GB]
    //   channel_density = 16 Gb @ 16단 (JEDEC 32ch × 16Gb 상한), die 수에 선형(12단 → 12 Gb)
    //   stack 수 = num_gpus_a × HBM4_STACKS_PER_GPU (배포 8×8 = 64 stack)
    //   → 16단·64스택 = 32×16/8 × 64 = 64 GB × 64 = 4096 GB = **4.096 TB**.
    //   (문서 OPERATING_POINT §4.1 의 "4.40 TB" 는 산술 오기 — 64GB×64=4.096TB 가 맞음.)
    constexpr double HBM4_BW_PER_STACK_TBPS         = 2.0;  // config.py:114
    constexpr double HBM4_CHANNEL_DENSITY_GB_16HIGH = 16.0; // JEDEC 채널밀도 상한 (16단)

    // KV 저장 = FP8 (8비트). 고정. FP16 경로 없음.
    constexpr int    KV_BYTES_PER_ELEM = 1;  // FP8 = 1 byte
    constexpr int    KV_KV_FACTOR      = 2;  // K 와 V
}

// ─────────────────────────────────────────────────────────────────────────────
// 변수 — AI 모델 스펙. 고정 아님.
// ─────────────────────────────────────────────────────────────────────────────
struct ModelSpec {
    int num_layers;
    int hidden;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int ffn_intermediate;
    int weight_bytes_per_elem = 2;  // 가중치 정밀도 동적: FP8=1 / FP16=2 / FP32=4 (기본 FP16)

    // KV bytes/token (FP8 aggregate over layers). config.py:130-131.
    // = 2(K,V) × num_kv_heads × head_dim × 1(FP8) × num_layers
    long long kv_bytes_per_token() const {
        return (long long)substrate::KV_KV_FACTOR * num_kv_heads * head_dim
             * substrate::KV_BYTES_PER_ELEM * num_layers;
    }

    // Instance A 가중치 bytes (QKV + O projection, FP16). KV 와 같은 64 스택을 점유.
    // QKV params/layer  = hidden × (num_heads×head_dim + 2×num_kv_heads×head_dim)
    // O proj params/layer = hidden × hidden
    // OPERATING_POINT §4.1: Llama70B ≈ 24 GB. 모델이 커지면 함께 커진다.
    long long instance_a_weight_bytes() const {
        long long qkv   = (long long)hidden * (num_heads * head_dim + 2 * num_kv_heads * head_dim);
        long long oproj = (long long)hidden * hidden;
        return (qkv + oproj) * num_layers * weight_bytes_per_elem;  // 동적 정밀도
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 변수 — GPU 하드웨어 스펙. 고정 아님.
// ─────────────────────────────────────────────────────────────────────────────
struct HwSpec {
    double gpu_peak_tflops;  // dense FP16 peak, per-GPU
    double gpu_mfu;          // achieved fraction (0..1)
    int    num_gpus_a = 8;   // Instance A tensor-parallel GPU 수 — 변수(배포 8, 다른 구성 가능)
    int    num_gpus_b = 8;   // Instance B tensor-parallel GPU 수 — 변수(배포 8)

    // 유효 연산 처리량 (FLOP/s) on n GPUs. config.py:215,269.
    double peak_flops(int num_gpus) const {
        return gpu_peak_tflops * 1e12 * gpu_mfu * num_gpus;
    }

    // Instance A PIM 채널 합 (k_aggregate). pim_emulator:51. num_gpus_a 변수(배포 8 → 2048).
    int k_aggregate() const {
        return num_gpus_a * substrate::HBM4_STACKS_PER_GPU * substrate::PIM_CHANNELS_PER_STACK;
    }
};

} // namespace puls
