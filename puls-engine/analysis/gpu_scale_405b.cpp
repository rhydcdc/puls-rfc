// PULS-ENGINE 분석 드라이버 — "Instance A 의 GPU 수(num_gpus_a)를 어디까지 늘리면
// GPU-A·GPU-B·PIM 3자원 균형을 맞추면서 405B 를 수용(hbm_fits)하는가".
// core/derive 의 derive_operating_point + core/optime 의 op-time 함수만 사용(공식 발명 0).
// 칩 = B200(2200 TFLOPS, MFU 0.6), 모델 = Llama-3.1 405B, prefill=128, HBM 16단.
//
// 빌드:
//   g++ -std=c++17 -O2 -I .../puls-engine analysis/gpu_scale_405b.cpp \
//       core/optime.cpp core/derive.cpp -o build/gpu_scale_405b.exe
//
// 배경(spec.h / derive.cpp):
//   - num_gpus_a 를 키우면 동시에 세 가지가 바뀐다:
//       (1) PIM 채널 k_aggregate = num_gpus_a × 8 × 32  ↑  → t_pim ↓
//       (2) GPU-A peak_flops = tflops×mfu×num_gpus_a    ↑  → t_gpu_a ↓
//       (3) HBM 용량 = num_gpus_a × 8 stack × per-stack  ↑  → cap ↑
//   - num_gpus_b 는 FFN(Instance B) peak_flops 만 결정 → t_ffn.
//   - derive 는 매 호출마다 t_pim=t_ffn=t_gpu_a 를 (N_dec, ctx) 로 다시 푼다.

#include "core/derive.h"
#include "core/optime.h"
#include <cstdio>

using namespace puls;

namespace {

// 405B (Llama-3.1) — kv_heads=8 GQA. test_derive.cpp / max_model.cpp 와 동일.
const ModelSpec M405{/*layers*/126, /*hidden*/16384, /*heads*/128,
                     /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/53248};
const int PREFILL = 128;

HwSpec make_hw(int num_gpus_a, int num_gpus_b) {
    HwSpec hw{/*tflops*/2200.0, /*mfu*/0.6};
    hw.num_gpus_a = num_gpus_a;
    hw.num_gpus_b = num_gpus_b;
    return hw;
}

// 한 (num_gpus_a, num_gpus_b) 구성을 derive 해서 한 줄 인쇄.
void print_row(int ga, int gb, const DeriveOptions& opt) {
    HwSpec hw = make_hw(ga, gb);
    OperatingPoint op = derive_operating_point(M405, hw, PREFILL, opt);
    std::printf("  A=%-3d B=%-3d | ctx=%8.0f N_dec=%-5d X=%7.2fus | a_tb=%6.2f cap=%6.2f fits=%s\n",
                ga, gb, op.ctx_balance, op.decode_count_target, op.balance_time_us,
                op.instance_a_tb, op.hbm_capacity_tb, op.hbm_fits ? "YES" : "NO ");
}

// fits=true 최소 num_gpus_a 를 1 step 으로 정밀 탐색(num_gpus_b 고정 정책 람다).
// gb_policy(ga) 가 해당 ga 에서 쓸 num_gpus_b 를 돌려준다.
template <typename GbPolicy>
int min_gpus_a_for_fit(int ga_start, int ga_max, GbPolicy gb_policy,
                       const DeriveOptions& opt) {
    for (int ga = ga_start; ga <= ga_max; ++ga) {
        HwSpec hw = make_hw(ga, gb_policy(ga));
        OperatingPoint op = derive_operating_point(M405, hw, PREFILL, opt);
        if (op.hbm_fits) return ga;
    }
    return -1;
}

// 찾은 지점에서 세 op-time 을 직접 계산해 3자원 균형(spread)을 검증.
void verify_balance(const char* label, int ga, int gb, const DeriveOptions& opt) {
    HwSpec hw = make_hw(ga, gb);
    OperatingPoint op = derive_operating_point(M405, hw, PREFILL, opt);

    const long long sum_kv   = (long long)op.decode_count_target * (long long)op.ctx_balance;
    const int       batch    = op.decode_count_target + PREFILL;
    const long long pa_work  = (long long)PREFILL * (long long)op.ctx_balance;

    const double t_pim   = t_pim_us(sum_kv, hw);
    const double t_ffn   = t_ffn_us(batch, M405, hw);
    const double t_gpu_a = t_gpu_a_us(batch, pa_work, M405, hw);

    const double mx = (t_pim > t_ffn ? (t_pim > t_gpu_a ? t_pim : t_gpu_a)
                                     : (t_ffn > t_gpu_a ? t_ffn : t_gpu_a));
    const double mn = (t_pim < t_ffn ? (t_pim < t_gpu_a ? t_pim : t_gpu_a)
                                     : (t_ffn < t_gpu_a ? t_ffn : t_gpu_a));
    const double spread_pct = (mx - mn) / mx * 100.0;

    std::printf("  %-22s A=%-3d B=%-3d  t_pim=%8.3f  t_ffn=%8.3f  t_gpu_a=%8.3f us"
                " | spread=%.3f%%  fits=%s\n",
                label, ga, gb, t_pim, t_ffn, t_gpu_a, spread_pct,
                op.hbm_fits ? "YES" : "NO");
}

} // namespace

int main() {
    DeriveOptions opt; opt.hbm_stack_height = 16;  // 배포 16단

    std::printf("=== PULS-ENGINE GPU-scale 분석 : 405B 수용 위한 num_gpus_a 탐색 ===\n");
    std::printf("    모델=Llama-3.1 405B (L=126 h=16384 kv=8 ffn=53248), 칩=B200(2200TF, MFU0.6)\n");
    std::printf("    prefill=%d, HBM 16단. 기준 배포 = A=8/B=8.\n", PREFILL);
    std::printf("    HBM 용량 = num_gpus_a × 8 stack × 64GB(16단) → A=8 이면 4.10TB.\n\n");

    // 405B 기준점(A=8/B=8) 한 줄 — fits 여부 먼저 확인.
    std::printf("─────────────────────────────────────────────────────────────────────────────\n");
    std::printf("기준 배포 A=8/B=8:\n");
    print_row(8, 8, opt);

    // ── §1. num_gpus_a 만 스윕 (num_gpus_b = 8 고정) ────────────────────────────
    std::printf("\n─────────────────────────────────────────────────────────────────────────────\n");
    std::printf("§1. num_gpus_a 스윕 (num_gpus_b=8 고정) — A 만 키운다\n");
    std::printf("    A↑ → PIM 채널↑·GPU-A peak↑·HBM cap↑.  B(FFN) 는 그대로 → FFN 이 점점 병목.\n");
    std::printf("─────────────────────────────────────────────────────────────────────────────\n");
    const int sweep[] = {8, 16, 24, 32, 40, 48, 56, 64};
    for (int ga : sweep) print_row(ga, 8, opt);

    const int min_a_only = min_gpus_a_for_fit(8, 4096, [](int){ return 8; }, opt);
    if (min_a_only > 0)
        std::printf("  → A 만 키울 때 405B fits=YES 최소 num_gpus_a = %d  (B=8 고정)\n", min_a_only);
    else
        std::printf("  → A 만 키워서는 (탐색 상한 내) fits 안 됨\n");

    // ── §2. num_gpus_a · num_gpus_b 동시 스윕 (A=B 동일 증가) ───────────────────
    std::printf("\n─────────────────────────────────────────────────────────────────────────────\n");
    std::printf("§2. num_gpus_a = num_gpus_b 동시 스윕 — A·B 같이 키운다\n");
    std::printf("    A·B 동시 → PIM·GPU-A·HBM 뿐 아니라 FFN(B) 도 같이 빨라짐 → 균형 더 자연스러움.\n");
    std::printf("─────────────────────────────────────────────────────────────────────────────\n");
    for (int g : sweep) print_row(g, g, opt);

    const int min_ab = min_gpus_a_for_fit(8, 4096, [](int ga){ return ga; }, opt);
    if (min_ab > 0)
        std::printf("  → A=B 동시일 때 405B fits=YES 최소 num_gpus(A=B) = %d\n", min_ab);
    else
        std::printf("  → A=B 동시로도 (탐색 상한 내) fits 안 됨\n");

    // ── §3. 두 경우 비교 해석 (수치로 뒷받침) ──────────────────────────────────
    std::printf("\n─────────────────────────────────────────────────────────────────────────────\n");
    std::printf("§3. 비교 — A 만 vs A·B 동시 (균형점 이동)\n");
    std::printf("─────────────────────────────────────────────────────────────────────────────\n");
    // HBM 용량은 num_gpus_a 에만 의존하므로, 두 경우의 fits 최소 num_gpus_a 는 같다(아래 확인).
    // 차이는 '같은 num_gpus_a 에서의 ctx/N_dec/X(균형점)' 에 나타난다.
    if (min_a_only > 0 && min_ab > 0) {
        std::printf("  같은 num_gpus_a=%d 에서 두 정책 비교:\n", min_a_only);
        std::printf("   [A 만, B=8]   "); print_row(min_a_only, 8, opt);
        std::printf("   [A·B 동시]    "); print_row(min_a_only, min_a_only, opt);
        std::printf("  관찰: HBM cap 은 num_gpus_a 에만 의존 → 두 정책의 fits 최소 num_gpus_a 동일.\n");
        std::printf("        B=8 고정이면 FFN(t_ffn)이 안 줄어 균형이 t_ffn 에 묶임 → 같은 X 를 맞추려\n");
        std::printf("        ctx·N_dec 가 (A·B 동시 대비) 다른 값으로 도출된다(위 두 줄 ctx/N_dec 비교).\n");
    }

    // ── §4. 검증 — 찾은 최소 GPU 수에서 세 op-time 직접 계산(3자원 균형 확인) ─────
    std::printf("\n─────────────────────────────────────────────────────────────────────────────\n");
    std::printf("§4. 검증 — 찾은 최소 GPU 수에서 t_pim ≈ t_ffn ≈ t_gpu_a (3자원 균형) 확인\n");
    std::printf("    op-time 직접 계산: t_pim(N_dec×ctx), t_ffn(N_dec+P), t_gpu_a(N_dec+P, P×ctx)\n");
    std::printf("─────────────────────────────────────────────────────────────────────────────\n");
    verify_balance("기준 A=8/B=8",        8, 8, opt);
    if (min_a_only > 0) verify_balance("A만 최소(B=8)",  min_a_only, 8,         opt);
    if (min_ab    > 0) verify_balance("A·B 동시 최소",   min_ab,     min_ab,    opt);

    return 0;
}
