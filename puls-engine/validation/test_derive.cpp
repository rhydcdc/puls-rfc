// 검증 — ① 파라미터 산출(derive). CONTRACT.md §7.
// 앵커: Llama-3 70B + B200 → prefill 128 = (ctx≈100K, N_dec≈62, kv≈6.15M, ffn≈190),
//       prefill 256 = 2배 스케일(123, 12.3M), ctx_balance 는 prefill 불변.
#include "test_framework.h"
#include "core/derive.h"
#include <cstdio>

using namespace puls;

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6, /*gpus_a*/8, /*gpus_b*/8};

    OperatingPoint op = derive_operating_point(llama, b200, 128);
    std::printf("P=128: ctx=%.0f N_dec=%d kv=%lld ffn_batch=%d X=%.2fus a_tb=%.2f fits=%d\n",
                op.ctx_balance, op.decode_count_target, op.kv_operating_target,
                op.ffn_batch, op.balance_time_us, op.instance_a_tb, (int)op.hbm_fits);
    CHECK_REL(op.ctx_balance, 100000.0, 0.10, "ctx_balance ~100K (chip constant)");
    CHECK(op.decode_count_target >= 58 && op.decode_count_target <= 66, "N_dec ~62");
    CHECK_REL(op.kv_operating_target, 6150000.0, 0.15, "kv_target ~6.15M");
    CHECK(op.ffn_batch >= 178 && op.ffn_batch <= 202, "ffn_batch ~190");
    CHECK(op.decode_pool == 2 * op.decode_count_target + 10, "decode_pool = 2N+10");
    CHECK(op.node_min == 2 * op.decode_count_target, "node_min = 2N");
    CHECK(op.hbm_fits, "HBM fits at prefill 128");

    OperatingPoint op2 = derive_operating_point(llama, b200, 256);
    std::printf("P=256: ctx=%.0f N_dec=%d kv=%lld ffn_batch=%d a_tb=%.2f fits=%d\n",
                op2.ctx_balance, op2.decode_count_target, op2.kv_operating_target,
                op2.ffn_batch, op2.instance_a_tb, (int)op2.hbm_fits);
    CHECK_REL(op2.ctx_balance, 100000.0, 0.10, "ctx_balance prefill-invariant");
    CHECK(op2.decode_count_target >= 116 && op2.decode_count_target <= 130, "N_dec ~123 (2x)");
    CHECK_REL(op2.kv_operating_target, 12300000.0, 0.15, "kv_target ~12.3M (2x)");

    // 스케일 불변성: 256 의 N_dec ≈ 2× 128 의 N_dec
    CHECK_NEAR(op2.decode_count_target, 2 * op.decode_count_target, 6, "N_dec scales 2x with prefill");

    return puls_test::summary();
}
