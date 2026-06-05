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
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};

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

    // ── HBM4 적합성 — 가중치 포함 + 활성 2배치(2×N_dec+잉여) + 모델 스케일 ─────────
    std::printf("HBM 70B: weight=%.1fGB a_tb=%.2f decode_pool=%d(=2x%d+%d) kv/tok=%lld\n",
                llama.instance_a_weight_bytes() / 1e9, op.instance_a_tb,
                op.decode_pool, op.decode_count_target,
                op.decode_pool - 2 * op.decode_count_target, llama.kv_bytes_per_token());
    CHECK_REL(llama.instance_a_weight_bytes() / 1e9, 24.0, 0.12, "Llama70B Instance A weight ~24GB");
    CHECK(op.decode_pool == 2 * op.decode_count_target + 10,
          "decode pool = 2 active microbatches (2xN_dec) + surplus");
    CHECK(op.hbm_fits, "70B fits with weights included");

    // 더 큰 모델(405B-class): KV/tok·가중치 모두 커져 HBM4 에 빠듯하게 적합(doc §4.1 ~4.24TB).
    ModelSpec m405{/*layers*/126, /*hidden*/16384, /*heads*/128,
                   /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/53248};
    OperatingPoint o405 = derive_operating_point(m405, b200, 128);
    std::printf("405B: ctx=%.0f N_dec=%d kv/tok=%.0fKiB weight=%.1fGB a_tb=%.2f fits=%d\n",
                o405.ctx_balance, o405.decode_count_target, m405.kv_bytes_per_token() / 1024.0,
                m405.instance_a_weight_bytes() / 1e9, o405.instance_a_tb, (int)o405.hbm_fits);
    CHECK_REL(m405.kv_bytes_per_token() / 1024.0, 252.0, 0.02, "405B KV/tok ~252 KiB (doc §4.1)");
    CHECK(o405.instance_a_tb > op.instance_a_tb, "bigger model uses more HBM (judged per model)");

    // ── HBM die-stack 높이 변수화 (12 vs 16단, 4-die SID 단위) ──────────────────
    DeriveOptions opt16; opt16.hbm_stack_height = 16;
    DeriveOptions opt12; opt12.hbm_stack_height = 12;
    OperatingPoint h16 = derive_operating_point(llama, b200, 128, opt16);
    OperatingPoint h12 = derive_operating_point(llama, b200, 128, opt12);
    OperatingPoint b16 = derive_operating_point(llama, b200, 192, opt16);  // 경계 데모
    OperatingPoint b12 = derive_operating_point(llama, b200, 192, opt12);
    std::printf("HBM stack: 16단 cap=%.2fTB | 12단 cap=%.2fTB | 70B@128 a_tb=%.2f | 70B@192 a_tb=%.2f(16:%d 12:%d)\n",
                h16.hbm_capacity_tb, h12.hbm_capacity_tb, h16.instance_a_tb,
                b16.instance_a_tb, (int)b16.hbm_fits, (int)b12.hbm_fits);
    // JEDEC 산출: 16단·64스택 = 32ch×16Gb/8 × 64 = 64GB×64 = 4096GB = 4.096 TB (문서 4.40 은 오기).
    CHECK_REL(h16.hbm_capacity_tb, 4.096, 0.005, "16단 = 4.096 TB (64stack × 64GB, JEDEC 32ch×16Gb)");
    CHECK_REL(h12.hbm_capacity_tb, 3.072, 0.005, "12단 = 3.072 TB (×12/16)");
    CHECK(h16.hbm_fits && h12.hbm_fits, "70B@128 (2.77TB) fits both 12/16단");
    // 경계: 12단(3.072)과 16단(4.096) 사이 점유면 16단만 적합 — die-stack 변수가 판정에 작용.
    if (b16.instance_a_tb > 3.072 && b16.instance_a_tb < 4.096) {
        CHECK(b16.hbm_fits && !b12.hbm_fits, "boundary load fits 16단 but not 12단 (변수 작용)");
    }

    return puls_test::summary();
}
