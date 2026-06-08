// proto/sim — 실 KPI 측정 (header-only). Plan2 CHECKLIST2 §2 kpi.h.
// TBT(=max(t_pim,t_gpu_a,t_ffn)) · TTFT · SLO goodput(출력토큰 가중) · idle-win(2자/3자).
// idle/Σdev 는 metrics.h(보조). 여기는 실 KPI(주 지표).
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "sim/harness.h"   // Deployment, OpTimes, round_optimes, round_us

namespace proto {

// SLO 임계 (가정 라벨, 스윕 가능).
struct SLO {
    double t_ttft_us;   // TTFT SLO
    double t_tbt_us;    // TBT(라운드) SLO
};

struct Kpi {
    std::vector<double> tbt;    // μ-batch-round 별 TBT 표본 (평균·p99)
    std::vector<double> ttft;   // 요청별 TTFT 표본 (큐 대기 + 프리필, 캐시-aware)

    // idle-win (가장 오래 idle = 가장 작은 op-time = 승)
    long long pim_win2 = 0, gpua_win2 = 0;                // 2자 GPU-A vs PIM (PIM 승률 = hiding)
    long long pim_win3 = 0, gpua_win3 = 0, ffn_win3 = 0;  // 3자
    // HBM 컨텐션 (Plan3): t_pim > t_gpu_a = PIM 노출. 노출 μ-batch 수 + 누적 노출시간(µs).
    long long mb_total = 0, exposed = 0;
    double    exposure_us = 0;   // Σ max(0, t_pim − t_gpu_a) × num_layers

    // SLO goodput (출력 토큰 = μ-batch decode_count 합)
    long long tokens_total   = 0;   // 전체 생성 토큰
    long long tokens_tbt_met = 0;   // TBT-SLO 충족 라운드의 토큰
    long long tokens_slo_met = 0;   // TBT-SLO ∧ (요청 TTFT-SLO 충족) 토큰
    long long reqs_total = 0, reqs_ttft_met = 0;
    double    sim_us = 0;            // goodput 분모(측정창 wall-time) — 드라이버가 (iters−warm)×round_us 로 세팅

    // μ-batch 1회 기록. ttft_met_count = 이 배치 내 *TTFT-SLO 충족 요청* 소속 디코더 수(드라이버가 셈).
    // beta = HBM 컨텐션 계수(Plan3): TBT = max(instance_A지연(β), t_ffn) × num_layers.
    //   instance_A지연 = max(pim,gpua) + β·max(0, pim−gpua). β=0 이면 기존 max(셋) 와 동일(회귀).
    void record_mubatch(int decode_count, long long sum_decode_kv, int ttft_met_count,
                        const Deployment& d, const SLO& slo, double beta) {
        const OpTimes t = round_optimes(decode_count, sum_decode_kv, d);
        // round_optimes 는 *레이어당* op-time → TBT(토큰간 = 전 레이어 forward-pass) = ... × num_layers.
        const double a_lat = instance_a_latency(t, beta);                 // A→B 의존성 + 컨텐션
        const double bt = std::max(a_lat, t.t_ffn) * (double)d.model.num_layers;
        tbt.push_back(bt);
        // 컨텐션 노출 (t_pim > t_gpu_a): 노출 μ-batch + 누적 노출시간.
        ++mb_total;
        const double exp = t.t_pim - t.t_gpu_a;
        if (exp > 0.0) { ++exposed; exposure_us += exp * (double)d.model.num_layers; }
        // idle-win: 작은 시간 = 더 오래 idle = 승.
        if (t.t_pim <= t.t_gpu_a) ++pim_win2; else ++gpua_win2;
        if (t.t_pim <= t.t_gpu_a && t.t_pim <= t.t_ffn) ++pim_win3;
        else if (t.t_gpu_a <= t.t_pim && t.t_gpu_a <= t.t_ffn) ++gpua_win3;
        else ++ffn_win3;
        // goodput
        tokens_total += decode_count;
        if (bt <= slo.t_tbt_us) {
            tokens_tbt_met += decode_count;
            tokens_slo_met += ttft_met_count;
        }
    }

    // 요청 1건 serve 시 TTFT 기록. 반환 = TTFT-SLO 충족 여부(드라이버가 디코더 met 플래그에 저장).
    bool record_ttft(double ttft_us, const SLO& slo) {
        ttft.push_back(ttft_us);
        ++reqs_total;
        const bool met = ttft_us <= slo.t_ttft_us;
        if (met) ++reqs_ttft_met;
        return met;
    }

    static double mean(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0; for (double x : v) s += x; return s / (double)v.size();
    }
    static double p99(std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t k = (size_t)std::llround(0.99 * (double)(v.size() - 1));
        return v[std::min(v.size() - 1, k)];
    }

    void print(const char* label, const SLO& slo) const {
        const double gp_tbt = sim_us > 0 ? (double)tokens_tbt_met / (sim_us / 1e6) : 0.0;
        const double gp_slo = sim_us > 0 ? (double)tokens_slo_met / (sim_us / 1e6) : 0.0;
        const long long w2 = pim_win2 + gpua_win2;
        const long long w3 = pim_win3 + gpua_win3 + ffn_win3;
        std::printf(
            "[KPI-%s] TBT mean=%.1f p99=%.1f us | TTFT mean=%.1f p99=%.1f us | "
            "PIMwin2=%.1f%% | win3 pim/gpua/ffn=%.0f/%.0f/%.0f%% | "
            "PIMexposed=%.1f%% expo_us=%.0f | "
            "goodput TBT=%.0f SLO=%.0f tok/s | TTFTmet=%.1f%% | T_ttft=%.0f T_tbt=%.1f\n",
            label, mean(tbt), p99(tbt), mean(ttft), p99(ttft),
            w2 ? 100.0 * (double)pim_win2 / (double)w2 : 0.0,
            w3 ? 100.0 * (double)pim_win3 / (double)w3 : 0.0,
            w3 ? 100.0 * (double)gpua_win3 / (double)w3 : 0.0,
            w3 ? 100.0 * (double)ffn_win3 / (double)w3 : 0.0,
            mb_total ? 100.0 * (double)exposed / (double)mb_total : 0.0,
            mb_total ? exposure_us / (double)mb_total : 0.0,
            gp_tbt, gp_slo,
            reqs_total ? 100.0 * (double)reqs_ttft_met / (double)reqs_total : 0.0,
            slo.t_ttft_us, slo.t_tbt_us);
    }
};

} // namespace proto
