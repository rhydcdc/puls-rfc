// PULS-ENGINE — 아키텍처 효율 분석 드라이버. (analysis, 코어 미터치.)
//
// 목적: 파이썬 evaluator.py 의 *집계* 아키텍처-효율 지표(idle_fraction,
// pipeline_efficiency, pim_utilization 류)를 C++ optime/derive 로 *동작점에서*
// 모델/HW 무관하게 일반화 산출. PIM 오프로드(t_pim≤t_gpuA)·A∥B 오버랩이
// idle≈0·PIM 숨음을 낸다는 아키텍처 주장(OPERATING_POINT §2)을 재현한다.
//
// *범위 밖*: 풀 이벤트-DAG sim, F1~F5 ablation(acceleration_decomposition 의
// F1~F5 는 event-sim/config 의존이라 제외). 여기서는 동작점 집계 지표만.
//
// 지표 정의는 evaluator.py 와 *정확히* 맞춘다:
//   - pipeline_efficiency = max(A,B)/(A+B)   (evaluator.py:228)  — 균형 A=B → 0.5
//   - PIM 숨음 = t_pim ≤ t_gpu_a             (OPERATING_POINT §2)
//   - inter-AB idle_fraction = |t_A−t_B| / max(t_A,t_B)  (덜 바쁜 인스턴스 유휴)
//   - 3자원 balance spread = (max−min)/max  over {t_pim, t_ffn, t_gpu_a}

#include "core/derive.h"
#include "core/optime.h"
#include "core/spec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace puls;

namespace {

// 동작점에서 산출한 효율 지표 묶음 (전부 derive 출력에서 일반화).
struct EffMetrics {
    double t_pim;       // Instance A attention (PIM)
    double t_gpu_a;     // Instance A proj + prefill-attn
    double t_ffn;       // Instance B FFN
    double t_A;         // Instance A 시간 = t_gpu_a (PIM 숨으면)
    double t_B;         // Instance B 시간 = t_ffn
    bool   pim_hidden;  // t_pim ≤ t_gpu_a ?
    double pim_margin;  // (t_gpu_a − t_pim) / t_gpu_a
    double pipeline_eff;    // max(A,B)/(A+B)         — evaluator.py 정의
    double idle_fraction;   // |A−B|/max(A,B)
    double balance_spread;  // (max−min)/max over 세 자원
};

EffMetrics compute(const ModelSpec& m, const HwSpec& hw, int prefill,
                   const DeriveOptions& opt = DeriveOptions{}) {
    OperatingPoint op = derive_operating_point(m, hw, prefill, opt);

    EffMetrics e{};
    e.t_ffn   = t_ffn_us(op.ffn_batch, m, hw);                                  // Instance B
    e.t_gpu_a = t_gpu_a_us(op.ffn_batch, op.prefill_kv_work_target, m, hw);     // Instance A proj+prefill-attn
    e.t_pim   = t_pim_us(op.kv_operating_target, hw);                           // Instance A attention(PIM)

    // PIM 숨음 (OP §2): t_pim 이 GPU-A 윈도우에 숨는가.
    e.pim_hidden = e.t_pim <= e.t_gpu_a;
    e.pim_margin = e.t_gpu_a > 0.0 ? (e.t_gpu_a - e.t_pim) / e.t_gpu_a : 0.0;

    // Instance A 시간 = t_gpu_a (PIM 숨으면), Instance B = t_ffn.
    e.t_A = e.t_gpu_a;
    e.t_B = e.t_ffn;

    // pipeline_efficiency = max(A,B)/(A+B) — evaluator.py:228 그대로.
    e.pipeline_eff = std::max(e.t_A, e.t_B) / (e.t_A + e.t_B);

    // inter-AB idle_fraction = |A−B|/max(A,B) — 덜 바쁜 인스턴스 유휴.
    const double mx_ab = std::max(e.t_A, e.t_B);
    e.idle_fraction = mx_ab > 0.0 ? std::fabs(e.t_A - e.t_B) / mx_ab : 0.0;

    // 3자원 balance spread = (max−min)/max over {t_pim, t_ffn, t_gpu_a}.
    const double mx3 = std::max({e.t_pim, e.t_ffn, e.t_gpu_a});
    const double mn3 = std::min({e.t_pim, e.t_ffn, e.t_gpu_a});
    e.balance_spread = mx3 > 0.0 ? (mx3 - mn3) / mx3 : 0.0;
    return e;
}

void print_case(const char* label, const ModelSpec& m, const HwSpec& hw,
                int prefill, const DeriveOptions& opt = DeriveOptions{}) {
    OperatingPoint op = derive_operating_point(m, hw, prefill, opt);
    EffMetrics e = compute(m, hw, prefill, opt);
    std::printf("=== %s (prefill=%d) ===\n", label, prefill);
    std::printf("  동작점: ctx=%.0f N_dec=%d ffn_batch=%d kv=%lld prefill_kv_work=%lld\n",
                op.ctx_balance, op.decode_count_target, op.ffn_batch,
                op.kv_operating_target, op.prefill_kv_work_target);
    std::printf("  세 자원(us): t_pim=%.2f  t_gpu_a=%.2f  t_ffn=%.2f\n",
                e.t_pim, e.t_gpu_a, e.t_ffn);
    std::printf("  PIM 숨음: t_pim<=t_gpu_a = %s  (%.2f <= %.2f, margin=%.4f)\n",
                e.pim_hidden ? "YES" : "NO", e.t_pim, e.t_gpu_a, e.pim_margin);
    std::printf("  Instance A=%.2f  B=%.2f\n", e.t_A, e.t_B);
    std::printf("  pipeline_efficiency = max(A,B)/(A+B) = %.4f  (균형 A=B -> 0.5)\n",
                e.pipeline_eff);
    std::printf("  inter-AB idle_fraction = |A-B|/max = %.4f  (균형 -> ~0)\n",
                e.idle_fraction);
    std::printf("  3자원 balance spread = (max-min)/max = %.4f  (~균형 -> ~0)\n\n",
                e.balance_spread);
}

} // namespace

int main() {
    // 앵커 — Llama-3 70B + B200, prefill 128 배포 동작점 (test_derive.cpp 와 동일 스펙).
    ModelSpec llama70b{/*layers*/80, /*hidden*/8192, /*heads*/64,
                       /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec    b200{/*tflops*/2200.0, /*mfu*/0.6};

    std::printf("# PULS 아키텍처 효율 — 동작점 집계 지표 (evaluator.py 정의 정합)\n");
    std::printf("# pipeline_efficiency=max(A,B)/(A+B); PIM 숨음=t_pim<=t_gpu_a (OP §2)\n\n");

    print_case("Llama-3 70B + B200", llama70b, b200, 128);

    // ── 일반화: 다른 모델/HW 에서도 동작점이 PIM 숨음·idle≈0·균형을 내는가 ──────
    // (1) 더 작은 모델 — Llama-3 8B 급 (hidden·layers·ffn 모두 작음).
    ModelSpec llama8b{/*layers*/32, /*hidden*/4096, /*heads*/32,
                      /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/14336};
    print_case("Llama-3 8B + B200", llama8b, b200, 128);

    // (2) 다른 HW 동작점 — 낮은 MFU(0.4) 의 B200 (HW 무관 일반화 확인).
    HwSpec b200_mfu04{/*tflops*/2200.0, /*mfu*/0.4};
    print_case("Llama-3 70B + B200(mfu=0.4)", llama70b, b200_mfu04, 128);

    return 0;
}
