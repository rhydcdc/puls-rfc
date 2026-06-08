// proto/validation — test_kpi. CHECKLIST2 §2 kpi.h 4 체크:
//  ① TBT = max(t_pim,t_gpu_a,t_ffn) × num_layers (per-layer 아님, 전 forward-pass).
//  ② idle-win: 작은 op-time = 더 오래 idle = 승 (큰 Σkv → t_pim 병목 → gpua 승, 작은 Σkv → pim 승).
//  ③ SLO goodput 이 TBT 임계 경계서 정확히 분기 (tokens_total/tbt_met/slo_met, ttft_met_count 전파).
//  ④ TTFT record + p99/mean 계산 정확.
#include "sim/kpi.h"

#include <cmath>
#include <cstdio>
#include <vector>

using proto::Deployment;
using proto::instance_a_latency;
using proto::Kpi;
using proto::make_deployment;
using proto::OpTimes;
using proto::round_optimes;
using proto::SLO;

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);\
            return 1;                                                     \
        }                                                                 \
    } while (0)

// 상대오차 1e-6 비교.
static bool near_rel(double a, double b) {
    const double denom = std::max(1.0, std::fabs(b));
    return std::fabs(a - b) / denom <= 1e-6;
}

// 상대오차 1e-9 비교 (컨텐션 TBT 식 정밀 검증용, CHECKLIST3 §4).
static bool near_rel9(double a, double b) {
    const double denom = std::max(1.0, std::fabs(b));
    return std::fabs(a - b) / denom <= 1e-9;
}

// ── ① TBT = max(셋) × num_layers ───────────────────────────────────────────
static int test_tbt_full_forward_pass() {
    Deployment d = make_deployment();
    const auto& op = d.op;
    const int dc = op.decode_count_target;  // 62

    // round_optimes 를 직접 호출해 expected 를 손계산.
    const OpTimes t = round_optimes(dc, op.kv_operating_target, d);
    const double mx = std::max(t.t_pim, std::max(t.t_gpu_a, t.t_ffn));
    const double expected = mx * (double)d.model.num_layers;

    SLO slo{5.0e6, 1.0e9};  // 임계 충분히 커서 분기 무관.
    Kpi k;
    k.record_mubatch(dc, op.kv_operating_target, dc, d, slo, 0.0);
    CHECK(k.tbt.size() == 1, "one tbt sample recorded");
    CHECK(near_rel(k.tbt[0], expected), "TBT == max(t_pim,t_gpu_a,t_ffn) * num_layers");
    // per-layer(×layers 없음)와 명확히 구별되는지 확인 (num_layers=80 → 80배 차이).
    CHECK(!near_rel(k.tbt[0], mx), "TBT is NOT the per-layer op-time (must be * num_layers)");
    return 0;
}

// ── ② idle-win: 작은 op-time 이 승 ─────────────────────────────────────────
static int test_idle_win_smaller_time_wins() {
    Deployment d = make_deployment();
    const auto& op = d.op;
    const int dc = op.decode_count_target;
    SLO slo{5.0e6, 1.0e9};

    // LARGE Σdecode_kv → t_pim 이 MAX(=PIM 병목) → PIM 은 idle 가장 적음 → 2자 패배(gpua 승).
    {
        Kpi k;
        const long long big = op.kv_operating_target * 3;
        // 사전 확인: 이 시나리오에서 t_pim 이 실제로 t_gpu_a 보다 크다(작은게 이긴다는 규칙의 전제).
        const OpTimes t = round_optimes(dc, big, d);
        CHECK(t.t_pim > t.t_gpu_a, "large Sigma-kv makes t_pim the bottleneck (> t_gpu_a)");
        k.record_mubatch(dc, big, dc, d, slo, 0.0);
        CHECK(k.gpua_win2 == 1 && k.pim_win2 == 0,
              "bottleneck PIM (largest time) loses 2-way; gpua (more idle) wins");
    }
    // SMALL Σdecode_kv → t_pim 이 최소 → PIM 이 가장 오래 idle → 2자 승.
    {
        Kpi k;
        const long long small = (long long)(op.kv_operating_target * 0.3);
        const OpTimes t = round_optimes(dc, small, d);
        CHECK(t.t_pim < t.t_gpu_a, "small Sigma-kv makes t_pim the smallest (< t_gpu_a)");
        k.record_mubatch(dc, small, dc, d, slo, 0.0);
        CHECK(k.pim_win2 == 1 && k.gpua_win2 == 0,
              "smallest time (PIM) wins 2-way (longest idle)");
    }
    return 0;
}

// ── ③ SLO goodput 이 TBT 임계 경계서 분기 ──────────────────────────────────
static int test_goodput_tbt_branch() {
    Deployment d = make_deployment();
    const auto& op = d.op;
    const int dc = op.decode_count_target;  // 62

    // on-target TBT 과 overloaded(3x) TBT 를 손계산.
    const OpTimes on = round_optimes(dc, op.kv_operating_target, d);
    const double tbt_on = std::max(on.t_pim, std::max(on.t_gpu_a, on.t_ffn))
                          * (double)d.model.num_layers;
    const OpTimes ov = round_optimes(dc, op.kv_operating_target * 3, d);
    const double tbt_ov = std::max(ov.t_pim, std::max(ov.t_gpu_a, ov.t_ffn))
                          * (double)d.model.num_layers;
    CHECK(tbt_ov > tbt_on, "overloaded TBT exceeds on-target TBT");

    // 임계를 on-target 바로 위 · overloaded 바로 아래로.
    SLO slo{5.0e6, (tbt_on + tbt_ov) * 0.5};
    CHECK(tbt_on <= slo.t_tbt_us && tbt_ov > slo.t_tbt_us, "threshold straddles the two TBTs");

    Kpi k;
    // (a) on-target: tbt 충족, ttft_met_count=62 → total+62, tbt_met+62, slo_met+62.
    k.record_mubatch(dc, op.kv_operating_target, dc, d, slo, 0.0);
    CHECK(k.tokens_total == dc, "on-target: tokens_total += decode_count");
    CHECK(k.tokens_tbt_met == dc, "on-target: tokens_tbt_met += decode_count");
    CHECK(k.tokens_slo_met == dc, "on-target: tokens_slo_met += ttft_met_count(=62)");

    // (b) overloaded: tbt 미충족 → total+62 만, tbt_met·slo_met 불변.
    k.record_mubatch(dc, op.kv_operating_target * 3, dc, d, slo, 0.0);
    CHECK(k.tokens_total == 2 * dc, "overloaded: tokens_total still increments");
    CHECK(k.tokens_tbt_met == dc, "overloaded: tokens_tbt_met unchanged (TBT > threshold)");
    CHECK(k.tokens_slo_met == dc, "overloaded: tokens_slo_met unchanged");

    // (c) ttft_met_count < decode_count 가 slo_met 에 전파 (tbt 충족일 때 met 수만 누적).
    const int met = 40;
    k.record_mubatch(dc, op.kv_operating_target, met, d, slo, 0.0);
    CHECK(k.tokens_total == 3 * dc, "tokens_total += decode_count regardless of ttft");
    CHECK(k.tokens_tbt_met == 2 * dc, "tokens_tbt_met += decode_count (TBT met)");
    CHECK(k.tokens_slo_met == dc + met, "tokens_slo_met += ttft_met_count(=40), not decode_count");
    return 0;
}

// ── ④ TTFT record + p99/mean ───────────────────────────────────────────────
static int test_ttft_record_and_percentiles() {
    SLO slo{50.5, 1.0e9};  // ttft ≤ 50.5 충족.
    Kpi k;
    // {1,2,...,100} 기록. met = ttft ≤ 50.5 → 1..50 충족(50건), 51..100 미충족.
    int expected_met = 0;
    for (int i = 1; i <= 100; ++i) {
        const bool met = k.record_ttft((double)i, slo);
        if (i <= 50) {
            CHECK(met, "ttft <= T_ttft returns met=true");
            ++expected_met;
        } else {
            CHECK(!met, "ttft > T_ttft returns met=false");
        }
    }
    CHECK(k.reqs_total == 100, "reqs_total counts every record_ttft");
    CHECK(k.reqs_ttft_met == expected_met && expected_met == 50, "reqs_ttft_met counts only met");

    // mean({1..100}) = 50.5.
    CHECK(near_rel(Kpi::mean(k.ttft), 50.5), "mean of 1..100 == 50.5");

    // p99: k = llround(0.99*(100-1)) = llround(98.01) = 98 → 정렬 vec[98] = 99.
    CHECK(near_rel(Kpi::p99(k.ttft), 99.0), "p99 of 1..100 == 99 (index 98)");

    // 빈 벡터 경계.
    std::vector<double> empty;
    CHECK(Kpi::mean(empty) == 0.0, "mean of empty == 0");
    CHECK(Kpi::p99(empty) == 0.0, "p99 of empty == 0");
    return 0;
}

// ── CHECKLIST3 §4 — HBM 컨텐션 모델 (Plan3) ─────────────────────────────────
static int test_contention_model() {
    Deployment d = make_deployment();
    const auto& op = d.op;
    const int dc = op.decode_count_target;       // 62
    const int L  = d.model.num_layers;            // 80
    SLO slo{5.0e6, 1.0e9};                         // TBT 임계 충분히 커서 분기 무관.

    // under/on-target: SMALL Σkv → t_pim ≤ t_gpu_a (PIM 이 GPU-A 그림자에 숨음).
    const long long under = (long long)(op.kv_operating_target * 0.1);
    const OpTimes tu = round_optimes(dc, under, d);
    CHECK(tu.t_pim <= tu.t_gpu_a, "under batch: t_pim <= t_gpu_a (PIM hidden)");

    // overloaded: 1.5× Σkv → t_pim > t_gpu_a (PIM 노출).
    const long long over = (long long)(op.kv_operating_target * 1.5);
    const OpTimes to = round_optimes(dc, over, d);
    CHECK(to.t_pim > to.t_gpu_a, "over batch: t_pim > t_gpu_a (PIM exposed)");

    // ── ① t_pim ≤ t_gpu_a ⇒ 페널티 0, β 무관 ───────────────────────────────
    // β=0 과 β=1.0 의 TBT 가 동일해야 하고, = max(t_gpu_a, t_ffn) × L.
    {
        Kpi k0, k1;
        k0.record_mubatch(dc, under, dc, d, slo, 0.0);
        k1.record_mubatch(dc, under, dc, d, slo, 1.0);
        const double expected = std::max(tu.t_gpu_a, tu.t_ffn) * (double)L;
        CHECK(near_rel9(k0.tbt.back(), k1.tbt.back()),
              "hidden PIM: TBT identical for beta=0 and beta=1");
        CHECK(near_rel9(k0.tbt.back(), expected),
              "hidden PIM: TBT == max(t_gpu_a, t_ffn) * num_layers");
        // 노출 카운터 미증가 (max(0,t_pim−t_gpu_a)=0).
        CHECK(k0.exposed == 0 && k1.exposed == 0, "hidden PIM: exposed counter not incremented");
        CHECK(k0.exposure_us == 0.0 && k1.exposure_us == 0.0, "hidden PIM: exposure_us stays 0");
    }

    // ── ② t_pim > t_gpu_a ⇒ TBT = max(max(pim,gpua)+β·diff, ffn) × L ────────
    const double diff = to.t_pim - to.t_gpu_a;     // > 0
    for (double beta : {0.5, 1.0}) {
        Kpi k;
        k.record_mubatch(dc, over, dc, d, slo, beta);
        const double a_lat = std::max(to.t_pim, to.t_gpu_a) + beta * diff;
        const double expected_tbt = std::max(a_lat, to.t_ffn) * (double)L;
        CHECK(near_rel9(k.tbt.back(), expected_tbt),
              "exposed PIM: TBT == max(max(pim,gpua)+beta*diff, ffn) * num_layers");
        // instance_a_latency 헬퍼와도 정합.
        CHECK(near_rel9(a_lat, instance_a_latency(to, beta)),
              "a_lat matches instance_a_latency(t, beta)");
        // 노출 카운터 증가 + 누적 노출시간 = diff × L.
        CHECK(k.exposed == 1, "exposed PIM: exposed counter incremented");
        CHECK(near_rel9(k.exposure_us, diff * (double)L),
              "exposed PIM: exposure_us += (t_pim - t_gpu_a) * num_layers");
    }

    // ── ③ β=0 회귀: TBT = max(t_pim,t_gpu_a,t_ffn) × L (옛 pre-Plan3 식) ────
    {
        Kpi k;
        k.record_mubatch(dc, over, dc, d, slo, 0.0);
        const double old_formula =
            std::max(to.t_pim, std::max(to.t_gpu_a, to.t_ffn)) * (double)L;
        CHECK(near_rel9(k.tbt.back(), old_formula),
              "beta=0 regression: TBT == max(t_pim,t_gpu_a,t_ffn) * num_layers (old formula)");
    }

    // ── ④ reload BW = SSD-class (SimConfig 기본값 교정) ─────────────────────
    // 100K-token KV(@kv_bytes_per_token)는 len*kv_bpt/2e7 라운드 = 수백 라운드(현실적)에 재로드.
    {
        proto::SimConfig cfg;
        CHECK(cfg.offload_bw_bytes_per_round == 2.0e7,
              "SimConfig default offload_bw is SSD-class (2e7 B/round)");
    }
    return 0;
}

int main() {
    if (test_tbt_full_forward_pass()) { std::fprintf(stderr, "test_tbt_full_forward_pass failed\n"); return 1; }
    if (test_idle_win_smaller_time_wins()) { std::fprintf(stderr, "test_idle_win_smaller_time_wins failed\n"); return 1; }
    if (test_goodput_tbt_branch()) { std::fprintf(stderr, "test_goodput_tbt_branch failed\n"); return 1; }
    if (test_ttft_record_and_percentiles()) { std::fprintf(stderr, "test_ttft_record_and_percentiles failed\n"); return 1; }
    if (test_contention_model()) { std::fprintf(stderr, "test_contention_model failed\n"); return 1; }
    std::printf("test_kpi OK\n");
    return 0;
}
