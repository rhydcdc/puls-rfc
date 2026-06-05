// PULS-ENGINE — op-time 모델 구현. CONTRACT.md §6 수식 박제.
// 단위 = 마이크로초(us). FLOPs 는 매우 크므로 정수 오버플로 방지를 위해 double 사용.
#include "core/optime.h"

namespace puls {

// PIM (Instance A attention). CONTRACT §6 / pim_emulator:53-88.
//   k_channels = num_gpus_a × HBM4_STACKS_PER_GPU × PIM_CHANNELS_PER_STACK
//   tiles      = ceil(sum_decode_kv_tokens / (k_channels × PIM_TILE_ROWS))
//   t_pim_us   = (tiles × PIM_TILE_TIME_FP8_NS + PIM_BROADCAST_NS) / 1000
double t_pim_us(long long sum_decode_kv_tokens, const HwSpec& hw) {
    if (sum_decode_kv_tokens <= 0) {
        return 0.0;  // decode rows 0 — pure-prefill batch
    }
    long long k_channels = (long long)hw.k_aggregate();
    long long denom = k_channels * (long long)substrate::PIM_TILE_ROWS;
    // ceil(a / b) for positive integers = (a + b - 1) / b
    long long tiles = (sum_decode_kv_tokens + denom - 1) / denom;
    return (tiles * substrate::PIM_TILE_TIME_FP8_NS + substrate::PIM_BROADCAST_NS) / 1000.0;
}

// FFN (Instance B). CONTRACT §6 / config.py:270-274.
//   flops    = 6 × batch_total × hidden × ffn_intermediate
//   t_ffn_us = flops / peak_flops(num_gpus_b) × 1e6
double t_ffn_us(int batch_total, const ModelSpec& m, const HwSpec& hw) {
    double flops = 6.0 * (double)batch_total * (double)m.hidden * (double)m.ffn_intermediate;
    return flops / hw.peak_flops(hw.num_gpus_b) * 1e6;
}

// GPU-A (A proj + prefill-attn). CONTRACT §6 / config.py:221-238,249.
//   qkv          = 2 × batch_total × hidden × (hidden + 2 × num_kv_heads × head_dim)
//   o_proj       = 2 × batch_total × hidden × hidden
//   prefill_attn = 2 × prefill_attn_work_tokens × hidden
//   t_gpu_a_us   = (qkv + o_proj + prefill_attn) / peak_flops(num_gpus_a) × 1e6
double t_gpu_a_us(int batch_total, long long prefill_attn_work_tokens,
                  const ModelSpec& m, const HwSpec& hw) {
    double qkv = 2.0 * (double)batch_total * (double)m.hidden
               * ((double)m.hidden + 2.0 * (double)m.num_kv_heads * (double)m.head_dim);
    double o_proj = 2.0 * (double)batch_total * (double)m.hidden * (double)m.hidden;
    double prefill_attn = 2.0 * (double)prefill_attn_work_tokens * (double)m.hidden;
    return (qkv + o_proj + prefill_attn) / hw.peak_flops(hw.num_gpus_a) * 1e6;
}

} // namespace puls
