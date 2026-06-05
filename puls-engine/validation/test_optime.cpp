// 검증 — op-time 박제 수식 ↔ 손계산 일치. CONTRACT.md §6.
// 앵커: Llama-3 70B(80,8192,64,8,128,28672) + B200(2200,0.6,8,8).
//   peak_flops(8) = 2200e12 × 0.6 × 8 = 1.056e16 FLOP/s  (a 와 b 동일: gpus 8)
//   k_channels    = 8 × 8 × 32 = 2048,  denom = 2048 × 32 = 65536
#include "test_framework.h"
#include "core/optime.h"
#include <cstdio>

using namespace puls;

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6, /*gpus_a*/8, /*gpus_b*/8};

    // ── FFN ──────────────────────────────────────────────────────────────────
    // flops = 6 × 190 × 8192 × 28672 = 267,852,840,960
    // t     = 267852840960 / 1.056e16 × 1e6 = 25.356474 us
    double ffn = t_ffn_us(190, llama, b200);
    std::printf("t_ffn_us(190)        = %.6f us  (expect 25.356474)\n", ffn);
    CHECK_REL(ffn, 25.356474, 0.02, "FFN op-time matches hand calc");

    // ── GPU-A ────────────────────────────────────────────────────────────────
    // batch_total = 190, prefill_attn_work_tokens = 128 × 100000 = 12,800,000
    //   qkv          = 2 × 190 × 8192 × (8192 + 2×8×128)  =  31,876,710,400
    //   o_proj       = 2 × 190 × 8192 × 8192              =  25,501,368,320
    //   prefill_attn = 2 × 12,800,000 × 8192             = 209,715,200,000
    //   sum = 267,093,278,720 / 1.056e16 × 1e6 = 25.292924 us
    double gpu_a = t_gpu_a_us(190, 12800000LL, llama, b200);
    std::printf("t_gpu_a_us(190, 12.8M)= %.6f us  (expect 25.292924)\n", gpu_a);
    CHECK_REL(gpu_a, 25.292924, 0.02, "GPU-A op-time matches hand calc");

    // ── PIM ──────────────────────────────────────────────────────────────────
    // sum_decode_kv_tokens = 62 × 100000 = 6,200,000
    //   tiles = ceil(6200000 / 65536) = ceil(94.604) = 95
    //   t     = (95 × 267 + 0.5) / 1000 = 25.3655 us
    double pim = t_pim_us(6200000LL, b200);
    std::printf("t_pim_us(6.2M)        = %.6f us  (expect 25.365500, tiles=95)\n", pim);
    CHECK_REL(pim, 25.365500, 0.02, "PIM op-time matches hand calc");
    // tiles 경계 재구성: round(t×1000 - 0.5)/267 == 95
    CHECK_NEAR((pim * 1000.0 - substrate::PIM_BROADCAST_NS) / substrate::PIM_TILE_TIME_FP8_NS,
               95.0, 1e-6, "PIM tile count = 95");

    // ── 경계: 빈 배치 (decode rows 0 — pure-prefill) ─────────────────────────
    // optime.cpp: sum_decode_kv_tokens <= 0 → 0.0 (early return, broadcast 미포함).
    double pim0 = t_pim_us(0, b200);
    std::printf("t_pim_us(0)           = %.6f us  (expect 0.0, empty batch)\n", pim0);
    CHECK_NEAR(pim0, 0.0, 1e-9, "PIM empty batch = 0 us");
    CHECK_NEAR(t_pim_us(-5, b200), 0.0, 1e-9, "PIM negative guarded to 0 us");

    // ── 단조성: 배치 ↑ → 시간 ↑ (비감소) ──────────────────────────────────
    // PIM: tile 단위 계단 함수 → 비감소.
    long long prev_in = 1;
    double prev_t = t_pim_us(prev_in, b200);
    bool pim_mono = true;
    long long pim_pts[] = {65536, 131072, 1000000, 6200000, 12300000};
    for (long long s : pim_pts) {
        double t = t_pim_us(s, b200);
        if (t < prev_t - 1e-12) pim_mono = false;
        prev_t = t;
    }
    CHECK(pim_mono, "PIM monotone non-decreasing in sum_decode_kv_tokens");

    // FFN: batch_total 선형 → 엄격 증가.
    bool ffn_mono = true;
    double pf = t_ffn_us(1, llama, b200);
    int ffn_pts[] = {2, 10, 100, 190, 256, 512};
    for (int b : ffn_pts) {
        double t = t_ffn_us(b, llama, b200);
        if (t <= pf) ffn_mono = false;
        pf = t;
    }
    CHECK(ffn_mono, "FFN strictly increasing in batch_total");

    // GPU-A: batch_total · prefill_work 둘 다 ↑ → 시간 ↑.
    bool gpu_mono = true;
    double pg = t_gpu_a_us(1, 0, llama, b200);
    long long work = 0;
    int gpu_pts[] = {10, 100, 190, 256};
    for (int b : gpu_pts) {
        work += 100000;
        double t = t_gpu_a_us(b, work, llama, b200);
        if (t <= pg) gpu_mono = false;
        pg = t;
    }
    CHECK(gpu_mono, "GPU-A increasing in (batch_total, prefill_work)");

    return puls_test::summary();
}
