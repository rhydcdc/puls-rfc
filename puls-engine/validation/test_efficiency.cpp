// 검증 — 아키텍처 효율(efficiency.cpp 와 동일 지표). analysis 드라이버의 동작점
// 집계 지표가 evaluator.py 정의대로이고, idle≈0·PIM 숨음·세 자원 균형이 여러
// 모델/HW 에서 일반화됨을 CHECK 한다. (코어 미터치 — 실패하면 사실대로 보고.)
//
// 지표 정의 (evaluator.py / OPERATING_POINT §2):
//   pipeline_efficiency = max(A,B)/(A+B)     (evaluator.py:228) — 균형 A=B → 0.5
//   PIM 숨음            = t_pim ≤ t_gpu_a     (OP §2)
//   inter-AB idle       = |t_A−t_B| / max(t_A,t_B)
//   balance spread      = (max−min)/max over {t_pim, t_ffn, t_gpu_a}

#include "test_framework.h"
#include "core/derive.h"
#include "core/optime.h"
#include "core/spec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace puls;

namespace {

struct EffMetrics {
    double t_pim, t_gpu_a, t_ffn, t_A, t_B;
    bool   pim_hidden;
    double pim_margin, pipeline_eff, idle_fraction, balance_spread;
};

EffMetrics compute(const ModelSpec& m, const HwSpec& hw, int prefill) {
    OperatingPoint op = derive_operating_point(m, hw, prefill);
    EffMetrics e{};
    e.t_ffn   = t_ffn_us(op.ffn_batch, m, hw);
    e.t_gpu_a = t_gpu_a_us(op.ffn_batch, op.prefill_kv_work_target, m, hw);
    e.t_pim   = t_pim_us(op.kv_operating_target, hw);
    e.pim_hidden = e.t_pim <= e.t_gpu_a;
    e.pim_margin = (e.t_gpu_a - e.t_pim) / e.t_gpu_a;
    e.t_A = e.t_gpu_a;  // PIM 숨으면 Instance A = GPU-A
    e.t_B = e.t_ffn;
    e.pipeline_eff  = std::max(e.t_A, e.t_B) / (e.t_A + e.t_B);          // evaluator.py:228
    e.idle_fraction = std::fabs(e.t_A - e.t_B) / std::max(e.t_A, e.t_B);
    const double mx3 = std::max({e.t_pim, e.t_ffn, e.t_gpu_a});
    const double mn3 = std::min({e.t_pim, e.t_ffn, e.t_gpu_a});
    e.balance_spread = (mx3 - mn3) / mx3;
    return e;
}

// 한 모델/HW 동작점에서 모든 효율 invariant 를 CHECK.
void check_case(const char* label, const ModelSpec& m, const HwSpec& hw, int prefill) {
    EffMetrics e = compute(m, hw, prefill);
    std::printf("[%s] t_pim=%.2f t_gpu_a=%.2f t_ffn=%.2f | margin=%.4f "
                "pipe_eff=%.4f idle=%.4f spread=%.4f\n",
                label, e.t_pim, e.t_gpu_a, e.t_ffn, e.pim_margin,
                e.pipeline_eff, e.idle_fraction, e.balance_spread);

    // PIM 숨음 (OP §2). derive 가 세 자원을 *co-equal* 로 풀므로 t_pim 은 t_gpu_a 에
    // 정수-라운딩(N_dec·ffn_batch·PIM tile ceil) 노이즈 내에서 올라앉는다. 따라서
    // 엄밀 t_pim≤t_gpu_a 는 라운딩에 따라 미소 초과할 수 있다. 라운딩 허용폭(2%)
    // 안에서 PIM 이 GPU-A 윈도우에 숨는지 검증 (margin ≥ −0.02).
    CHECK(e.pim_margin >= -0.02, "PIM hidden within GPU-A window (margin >= -2%, 라운딩 허용)");

    // inter-AB idle ≈ 0 (동작점 균형). 강한 주장 — 엄밀 ≤ 0.05.
    CHECK(e.idle_fraction <= 0.05, "inter-AB idle_fraction <= 0.05 (동작점 ~0)");

    // 세 자원 ≈ 균형. 강한 주장 — 엄밀 ≤ 0.05.
    CHECK(e.balance_spread <= 0.05, "3-resource balance spread <= 0.05");

    // pipeline_efficiency 정의 정합 — 균형이면 ≈ 0.5 (evaluator.py).
    CHECK_NEAR(e.pipeline_eff, 0.5, 0.05, "pipeline_efficiency ~= 0.5 (balanced A=B)");
}

} // namespace

int main() {
    ModelSpec llama70b{/*layers*/80, /*hidden*/8192, /*heads*/64,
                       /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    ModelSpec llama8b {/*layers*/32, /*hidden*/4096, /*heads*/32,
                       /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/14336};
    HwSpec    b200      {/*tflops*/2200.0, /*mfu*/0.6};
    HwSpec    b200_mfu04{/*tflops*/2200.0, /*mfu*/0.4};

    // 앵커 + 일반화(>=2 모델/HW): 70B, 8B, 70B@mfu0.4.
    check_case("Llama70B+B200",        llama70b, b200,       128);
    check_case("Llama8B+B200",         llama8b,  b200,       128);
    check_case("Llama70B+B200(mfu0.4)", llama70b, b200_mfu04, 128);

    // pipeline_efficiency 정의 단위 테스트 — evaluator.py 의 경계와 정합.
    // A=B → 0.5; A>>B → 1.0. (max(A,B)/(A+B))
    {
        auto pe = [](double a, double b){ return std::max(a,b)/(a+b); };
        CHECK_NEAR(pe(10.0, 10.0), 0.5, 1e-12, "pipeline_eff(A=B)=0.5");
        CHECK(pe(100.0, 1.0) > 0.98, "pipeline_eff(A>>B)->1.0 (pipeline 무의미)");
    }

    return puls_test::summary();
}
