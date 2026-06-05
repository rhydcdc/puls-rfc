// PULS-ENGINE — ① 파라미터 산출 구현. CONTRACT.md §3-① · §6 · §5.2.
// 문서가 손으로 푼 세 자원 균형(t_PIM = t_FFN = t_GPU-A)을 optime.h 의 실제 op-time
// 함수로 수치 해법(이분법)으로 푼다 — 공식 발명 0, 동작점 수치 리터럴 0(§9-5).
//
// 균형 미지수 = (N_dec, ctx). 조건:
//   t_pim_us (N_dec × ctx)            =  Σ decode KV → PIM tiles
//   t_ffn_us (N_dec + P)              =  batch 토큰 → FFN FLOPs
//   t_gpu_a_us(N_dec + P, P × ctx)    =  A proj + prefill-attn(Σ chunk×depth = P×ctx)
//
// 해법(2단 이분):
//   바깥: ctx 에 대해 이분. 각 ctx 에서 t_ffn = t_gpu_a 로 N_dec(ctx) 를 풀고,
//         잔차 r(ctx) = t_pim(N_dec×ctx) − t_ffn(N_dec+P) 의 부호 변화를 추적.
//   안쪽: 고정 ctx 에서 t_ffn(N+P) − t_gpu_a(N+P, P×ctx) 가 N 에 대해 단조 증가
//         (FFN flops/tok > A-proj flops/tok, 차는 ctx 무관 상수 prefill-attn 항).
//         N=0 에서 음수(prefill-attn 으로 GPU-A 가 큼) → 증가하며 0 통과. 그 근을 이분.
//   바깥 r(ctx): 작은 ctx 에서 PIM≪FFN(음수), 큰 ctx 에서 PIM≫FFN(양수) → 단조, 근 유일.

#include "derive.h"
#include "optime.h"

#include <cmath>

namespace puls {

namespace {

// 고정 ctx 에서 t_ffn(N+P) = t_gpu_a(N+P, P×ctx) 를 만족하는 실수 N_dec.
// f(N) = t_ffn − t_gpu_a 는 N 에 대해 단조 증가(f(0) ≤ 0). 그 근을 이분으로.
double ndec_ffn_eq_gpua(double ctx, int prefill_tokens,
                        const ModelSpec& m, const HwSpec& hw) {
    const long long prefill_attn_work = (long long)std::llround(prefill_tokens * ctx);
    auto f = [&](double n) {
        const int batch = (int)std::llround(n) + prefill_tokens;
        return t_ffn_us(batch, m, hw)
             - t_gpu_a_us(batch, prefill_attn_work, m, hw);
    };
    double lo = 0.0, hi = 1.0e6;
    if (f(lo) > 0.0) return 0.0;   // 균형 N_dec 없음(축퇴) — 0 으로 클램프
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) < 0.0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// 잔차 r(ctx) = t_pim(N_dec×ctx) − t_ffn(N_dec+P), N_dec 는 위 FFN=GPU-A 해.
double balance_residual(double ctx, int prefill_tokens,
                        const ModelSpec& m, const HwSpec& hw) {
    const double n = ndec_ffn_eq_gpua(ctx, prefill_tokens, m, hw);
    const long long sum_kv = (long long)std::llround(n * ctx);
    const int batch = (int)std::llround(n) + prefill_tokens;
    return t_pim_us(sum_kv, hw) - t_ffn_us(batch, m, hw);
}

} // namespace

OperatingPoint derive_operating_point(const ModelSpec& model,
                                      const HwSpec& hw,
                                      int prefill_tokens,
                                      const DeriveOptions& opt) {
    const int P = prefill_tokens;

    // ── 세 자원 균형: ctx 에 대해 이분(잔차 부호 변화 추적) ───────────────────
    double lo = 1.0e3, hi = 4.0e5;
    const bool neg_at_lo = balance_residual(lo, P, model, hw) < 0.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        const bool neg = balance_residual(mid, P, model, hw) < 0.0;
        if (neg == neg_at_lo) lo = mid; else hi = mid;
    }
    const double ctx = 0.5 * (lo + hi);
    const double n_dec_real = ndec_ffn_eq_gpua(ctx, P, model, hw);

    // ── 도출 결과 ─────────────────────────────────────────────────────────────
    const int n_dec = (int)std::llround(n_dec_real);

    OperatingPoint op{};
    op.ctx_balance            = ctx;
    op.decode_count_target    = n_dec;
    op.kv_operating_target    = (long long)std::llround((double)n_dec * ctx);
    op.prefill_tokens         = P;
    op.prefill_kv_work_target = (long long)std::llround((double)P * ctx);
    op.ffn_batch              = n_dec + P;
    op.balance_time_us        = t_ffn_us(op.ffn_batch, model, hw);

    // ── 풀 구성 ───────────────────────────────────────────────────────────────
    op.decode_pool  = 2 * n_dec + opt.decode_surplus;
    op.prefill_pool = opt.prefill_pool;
    op.age_cap      = opt.age_cap;
    op.idle_band    = opt.idle_band;

    // ── 클러스터 라우팅 경계 ──────────────────────────────────────────────────
    op.node_min   = 2 * n_dec;                       // 2 μ-batch 바닥
    op.node_max   = op.node_min + opt.node_max_surplus;
    op.edge_band  = opt.edge_band_tokens;
    // footprint 캡 = 2 μ-batch KV(2×kv_target) + 여유. count 상한(node_max)보다 헤드룸을 둬
    // 긴 꼬리 다양성을 보존(없으면 노드가 134 전에 막혀 on2↓). OP §4.1: 30M@256 / 15M@128.
    op.node_footprint_cap =
        (long long)std::llround(2.0 * (double)op.kv_operating_target * opt.footprint_headroom);

    // ── HBM 적합성 (CONTRACT §6 / OP §4.1) ────────────────────────────────────
    // Instance A footprint = 활성 2 μ-batch 디코드 풀(2×N_dec+잉여) + 프리필 in-flight 의
    // KV(FP8) + QKV/O proj 가중치(동적 정밀도). 가용 용량은 die-stack 높이에 비례.
    const double instance_a_tokens =
        op.decode_pool * ctx                                  // = (2×N_dec+잉여) × ctx
      + op.prefill_pool * (ctx * opt.prefill_avg_depth_frac); // 프리필 in-flight 평균 depth
    const double kv_bytes     = instance_a_tokens * (double)model.kv_bytes_per_token();
    const double weight_bytes = (double)model.instance_a_weight_bytes();  // QKV/O proj, 동적 정밀도
    op.instance_a_tb = (kv_bytes + weight_bytes) / 1e12;
    // 가용 HBM = 16단 기준(4.40 TB)을 die-stack 높이로 선형 스케일(4-die SID 단위, 4의 배수).
    op.hbm_capacity_tb = substrate::PIM_CAP_TB * (opt.hbm_stack_height / 16.0);
    op.hbm_fits        = op.instance_a_tb <= op.hbm_capacity_tb;

    return op;
}

} // namespace puls
