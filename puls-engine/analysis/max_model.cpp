// PULS-ENGINE 분석 드라이버 — "칩 성능별로 어떤 크기의 AI 모델까지 HBM 에 적합한가".
// core/derive 의 derive_operating_point 만 사용(공식 발명 0). 칩 = B200.
// HBM 적합성 경계(hbm_fits true→false)를 SOTA config 와 dense 스윕으로 산출한다.
//
// 빌드:
//   g++ -std=c++17 -O2 -I .../puls-engine max_model.cpp \
//       core/optime.cpp core/derive.cpp -o build/max_model.exe
//
// 주의: 파라미터수 = 12 × layers × hidden^2 는 트랜스포머 대략 근사(attn+ffn 합산의 통상
//       경험식). MoE(DeepSeek-V3 류)는 active 가 아니라 layer/kv_head 구조라 dense 식이
//       KV/가중치를 과대평가하므로 본 드라이버에서 의도적으로 생략한다(주석 §B).

#include "core/derive.h"
#include <cstdio>
#include <string>

using namespace puls;

namespace {

// 트랜스포머 파라미터수 대략 근사(십억=B 단위). 12 × L × hidden^2.
double approx_params_B(int layers, int hidden) {
    return 12.0 * (double)layers * (double)hidden * (double)hidden / 1e9;
}

// 한 모델 config 를 derive 해서 한 줄 인쇄.
void print_model(const char* name, const ModelSpec& m, const HwSpec& hw,
                 int prefill, const DeriveOptions& opt) {
    OperatingPoint op = derive_operating_point(m, hw, prefill, opt);
    std::printf("  %-16s L=%-4d h=%-6d kv_h=%-2d | fits=%-3s a_tb=%5.2f / cap=%4.2f"
                " | ctx=%6.0f N_dec=%-4d kv/tok=%4.0fKiB w=%5.1fGB\n",
                name, m.num_layers, m.hidden, m.num_kv_heads,
                op.hbm_fits ? "YES" : "NO", op.instance_a_tb, op.hbm_capacity_tb,
                op.ctx_balance, op.decode_count_target,
                m.kv_bytes_per_token() / 1024.0,
                m.instance_a_weight_bytes() / 1e9);
}

// 16단/12단 두 stack-height 로 같은 모델 인쇄(가독용 헤더 포함).
void print_model_both(const char* name, ModelSpec m, const HwSpec& hw, int prefill) {
    DeriveOptions o16; o16.hbm_stack_height = 16;
    DeriveOptions o12; o12.hbm_stack_height = 12;
    std::printf("[%s] FP16(w=2)\n", name);
    std::printf("  16단:"); print_model(name, m, hw, prefill, o16);
    std::printf("  12단:"); print_model(name, m, hw, prefill, o12);
    // FP8 가중치 비교 한 줄(KV 는 substrate 상 FP8 고정, 가중치만 1바이트).
    ModelSpec mf = m; mf.weight_bytes_per_elem = 1;
    OperatingPoint o = derive_operating_point(mf, hw, prefill, o16);
    std::printf("  FP8w 16단:  fits=%-3s a_tb=%5.2f (가중치 %.1fGB→%.1fGB, KV 는 FP8 고정)\n",
                o.hbm_fits ? "YES" : "NO", o.instance_a_tb,
                m.instance_a_weight_bytes() / 1e9, mf.instance_a_weight_bytes() / 1e9);
}

// dense 경계 스윕: Llama 비율(heads=hidden/128, kv_heads=8, head_dim=128,
// ffn≈3.5×hidden)을 유지하며 한 축을 키워 hbm_fits true→false 직전을 찾는다.
// mode==0: hidden 고정·layers 증가 / mode==1: layers 고정·hidden 증가(128 배수).
void sweep_boundary(const char* label, const HwSpec& hw, int prefill,
                    int stack_height, int mode, int layers0, int hidden0) {
    DeriveOptions opt; opt.hbm_stack_height = stack_height;

    auto make = [&](int layers, int hidden) {
        int heads    = hidden / 128;            // head_dim=128 가정
        int kv_heads = 8;                       // Llama GQA 고정
        int head_dim = 128;
        int ffn      = (int)(3.5 * hidden);     // Llama ffn≈3.5×hidden
        return ModelSpec{layers, hidden, heads, kv_heads, head_dim, ffn};
    };

    int last_L = -1, last_h = -1;
    double last_a_tb = 0, last_ctx = 0, last_cap = 0;
    int last_ndec = 0;
    for (int step = 0; step < 4000; ++step) {
        int layers = (mode == 0) ? (layers0 + step)        : layers0;
        int hidden = (mode == 1) ? (hidden0 + 128 * step)  : hidden0;
        if (mode == 1 && (hidden / 128) < 1) continue;
        ModelSpec m = make(layers, hidden);
        OperatingPoint op = derive_operating_point(m, hw, prefill, opt);
        if (!op.hbm_fits) break;                // true→false 전환: 직전이 최대
        last_L = layers; last_h = hidden;
        last_a_tb = op.instance_a_tb; last_ctx = op.ctx_balance;
        last_cap = op.hbm_capacity_tb; last_ndec = op.decode_count_target;
    }
    if (last_L < 0) { std::printf("  %-26s 시작점부터 불적합\n", label); return; }
    std::printf("  %-26s 최대: L=%-4d h=%-6d ~%.0fB params | ctx=%6.0f N_dec=%-3d"
                " a_tb=%5.2f/%4.2f\n",
                label, last_L, last_h, approx_params_B(last_L, last_h),
                last_ctx, last_ndec, last_a_tb, last_cap);
}

} // namespace

int main() {
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};   // num_gpus_a 기본 8
    const int prefill = 128;

    std::printf("=== PULS-ENGINE max-model 분석 : 칩=B200(2200 TFLOPS, MFU 0.6, 8 GPU), prefill=%d ===\n", prefill);
    std::printf("    HBM 가용(JEDEC 산출): 16단=4.10TB / 12단=3.07TB (die-stack 선형). KV=FP8 고정.\n\n");

    // ── §A. SOTA dense 모델 적합성 (16단 / 12단 / FP8 가중치) ────────────────────
    std::printf("─────────────────────────────────────────────────────────────────────\n");
    std::printf("§A. 실제 SOTA dense 모델 적합성\n");
    std::printf("─────────────────────────────────────────────────────────────────────\n");
    print_model_both("Llama-3 8B",    ModelSpec{32,  4096,  32, 8, 128, 14336}, b200, prefill);
    print_model_both("Llama-3 70B",   ModelSpec{80,  8192,  64, 8, 128, 28672}, b200, prefill);
    print_model_both("Llama-3.1 405B", ModelSpec{126, 16384, 128, 8, 128, 53248}, b200, prefill);
    print_model_both("Qwen2.5-72B",   ModelSpec{80,  8192,  64, 8, 128, 29568}, b200, prefill);
    // §B. MoE 한계: DeepSeek-V3(671B 총/37B active)류는 KV·가중치가 active 가 아니라
    // layer/kv_head 구조에서 나와 dense 식으로는 총-param 가중치를 과대평가한다. 또한
    // MLA(latent KV) 라 본 KV/tok 식과도 다르다 → 정직성 위해 생략.
    std::printf("  (MoE: DeepSeek-V3 류는 active≠구조라 dense 근사 부정확 + MLA KV → 생략, §B)\n");

    // ── §C. 최대 dense 모델 경계 스윕 (16단 / 12단) ─────────────────────────────
    std::printf("\n─────────────────────────────────────────────────────────────────────\n");
    std::printf("§C. 최대 dense 모델 경계 (Llama 비율 유지: heads=h/128, kv=8, ffn=3.5h)\n");
    std::printf("    params 근사 = 12 × layers × hidden^2 (트랜스포머 대략, 근사임)\n");
    std::printf("─────────────────────────────────────────────────────────────────────\n");
    std::printf(" [16단 = 4.10 TB]\n");
    sweep_boundary("hidden=8192 고정, L↑",  b200, prefill, 16, /*mode layers*/0, /*L0*/8,  /*h*/8192);
    sweep_boundary("hidden=16384 고정, L↑", b200, prefill, 16, 0, 8,  16384);
    sweep_boundary("layers=80 고정, h↑",    b200, prefill, 16, /*mode hidden*/1, /*L*/80, /*h0*/2048);
    sweep_boundary("layers=126 고정, h↑",   b200, prefill, 16, 1, 126, 2048);
    std::printf(" [12단 = 3.07 TB]\n");
    sweep_boundary("hidden=8192 고정, L↑",  b200, prefill, 12, 0, 8,  8192);
    sweep_boundary("hidden=16384 고정, L↑", b200, prefill, 12, 0, 8,  16384);
    sweep_boundary("layers=80 고정, h↑",    b200, prefill, 12, 1, 80, 2048);
    sweep_boundary("layers=126 고정, h↑",   b200, prefill, 12, 1, 126, 2048);

    // ── §D. 중요 관찰: ctx_balance 가 모델 크기와 함께 상승 ──────────────────────
    std::printf("\n─────────────────────────────────────────────────────────────────────\n");
    std::printf("§D. 관찰: ctx_balance 는 모델별로 재도출된다(고정 100K 아님)\n");
    std::printf("─────────────────────────────────────────────────────────────────────\n");
    DeriveOptions od; od.hbm_stack_height = 16;
    struct { const char* n; ModelSpec m; } row[] = {
        {"Llama-3 8B",    ModelSpec{32,  4096,  32, 8, 128, 14336}},
        {"Llama-3 70B",   ModelSpec{80,  8192,  64, 8, 128, 28672}},
        {"Llama-3.1 405B", ModelSpec{126, 16384, 128, 8, 128, 53248}},
    };
    for (auto& r : row) {
        OperatingPoint op = derive_operating_point(r.m, b200, prefill, od);
        std::printf("  %-16s ctx_balance=%7.0f  N_dec=%-4d  kv/tok=%4.0fKiB  a_tb=%.2f\n",
                    r.n, op.ctx_balance, op.decode_count_target,
                    r.m.kv_bytes_per_token() / 1024.0, op.instance_a_tb);
    }
    std::printf("  → 큰 모델일수록 KV/tok 이 layers·kv 양쪽으로 커져, 같은 4.10TB 안에서\n");
    std::printf("    버틸 수 있는 ctx·decode 수가 줄고 균형 ctx 가 재조정된다. 이는 스펙(용량)\n");
    std::printf("    한계이지 일반화 실패가 아니다 — derive 는 모델별 균형을 매번 다시 푼다.\n");

    return 0;
}
