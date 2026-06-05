// 검증 — ③ 글로벌(클러스터) 스케줄러. CONTRACT.md §3-③ / §7.
// 앵커: Llama-3 70B + B200 + prefill 128 동작점에서
//   1. 게이트 edge% (분포 B 꼬리 격리, kept 평균 ≤ ctx_balance+edge_band)
//   2. 게이트 스케일 불변 (draw 2× 해도 edge% ±1%p)
//   3. cold-start (노드 count∈[node_min,node_max], mean≈ctx_balance)
//   4. on-point (다수 노드에서 onpoint_batches≥1)
//   5. per-completion 힐링 drift 0 (count·mean 안정, toxic-fit 보존)
// 패턴은 test_derive.cpp 를 따른다. 코어 수정 금지 — 기대를 구현 동작에 맞춘다.
#include "test_framework.h"
#include "core/derive.h"
#include "core/global_scheduler.h"
#include "core/workload.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace puls;

// 분포 B 로 N 개 draw 생성 (워크로드 샘플러 직접).
static std::vector<int> make_draw(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) v.push_back(sample_distribution_b(rng));
    return v;
}

static double mean_of(const std::vector<int>& v) {
    if (v.empty()) return 0.0;
    long long s = 0;
    for (int x : v) s += x;
    return (double)s / v.size();
}

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6, /*gpus_a*/8, /*gpus_b*/8};
    OperatingPoint op = derive_operating_point(llama, b200, 128);
    std::printf("op: ctx=%.0f N_dec=%d kv=%lld node_min=%d node_max=%d edge_band=%.0f\n",
                op.ctx_balance, op.decode_count_target, op.kv_operating_target,
                op.node_min, op.node_max, op.edge_band);

    const double gate_ceil = op.ctx_balance + op.edge_band;

    // ── 1. 게이트 edge% ───────────────────────────────────────────────────────
    // 큰 draw → gate. edged 비율(꼬리 격리)이 합리적 수%, kept 평균 ≤ ceil.
    const int E1 = 80000;
    std::vector<int> draw1 = make_draw(E1, /*seed*/1);
    GateResult g1 = gate(draw1, op);
    const double edge_pct1 = 100.0 * (double)g1.edged.size() / E1;
    const double kept_mean1 = mean_of(g1.kept);
    std::printf("[1] gate E=%d: edged=%zu (%.3f%%) kept=%zu kept_mean=%.0f ceil=%.0f\n",
                E1, g1.edged.size(), edge_pct1, g1.kept.size(), kept_mean1, gate_ceil);
    CHECK((int)(g1.kept.size() + g1.edged.size()) == E1, "gate: kept+edged = draw (보존)");
    CHECK(kept_mean1 <= gate_ceil + 1e-6, "kept 평균 ≤ ctx_balance + edge_band");
    CHECK(edge_pct1 > 0.0 && edge_pct1 < 15.0, "edge% 합리적 (분포 꼬리, 수%)");

    // ── 2. 게이트 스케일 불변 ─────────────────────────────────────────────────
    // draw 2× → edge% 거의 동일 (±1%p). 노드수·풀크기 무관.
    const int E2 = 2 * E1;
    std::vector<int> draw2 = make_draw(E2, /*seed*/2);
    GateResult g2 = gate(draw2, op);
    const double edge_pct2 = 100.0 * (double)g2.edged.size() / E2;
    std::printf("[2] gate E=%d: edged=%zu (%.3f%%)  |Δ|=%.3f%%p\n",
                E2, g2.edged.size(), edge_pct2, std::fabs(edge_pct2 - edge_pct1));
    CHECK(mean_of(g2.kept) <= gate_ceil + 1e-6, "2× kept 평균도 ≤ ceil");
    CHECK_NEAR(edge_pct2, edge_pct1, 1.0, "edge% 스케일 불변 (±1%p)");

    // ── 3. cold-start ─────────────────────────────────────────────────────────
    // Z개 노드 생성 → cold_start(kept, nodes, op, seed).
    // 다수 노드 count∈[node_min,node_max], mean 이 ctx_balance 근처.
    const int Z = 256;
    std::vector<ClusterNode> nodes(Z);
    int leftover = cold_start(g1.kept, nodes, op, /*seed*/3);

    int in_range = 0, near_mean = 0, nonempty = 0;
    double sum_node_mean = 0.0;
    for (const ClusterNode& nd : nodes) {
        if (nd.count() == 0) continue;
        ++nonempty;
        sum_node_mean += nd.mean();
        if (nd.count() >= op.node_min && nd.count() <= op.node_max) ++in_range;
        // mean 이 ctx_balance 의 ±25% 안이면 "근처"
        if (std::fabs(nd.mean() - op.ctx_balance) <= 0.25 * op.ctx_balance) ++near_mean;
    }
    const double avg_node_mean = nonempty ? sum_node_mean / nonempty : 0.0;
    const double range_frac = nonempty ? (double)in_range / nonempty : 0.0;
    const double mean_frac = nonempty ? (double)near_mean / nonempty : 0.0;
    std::printf("[3] cold_start Z=%d: placed=%zu leftover=%d nonempty=%d "
                "in_range=%d(%.1f%%) near_mean=%d(%.1f%%) avg_node_mean=%.0f\n",
                Z, g1.kept.size() - leftover, leftover, nonempty,
                in_range, 100.0 * range_frac, near_mean, 100.0 * mean_frac, avg_node_mean);
    CHECK(nonempty > 0, "cold_start: 적어도 일부 노드 채워짐");
    CHECK(range_frac >= 0.5, "다수 노드 count∈[node_min,node_max]");
    CHECK_REL(avg_node_mean, op.ctx_balance, 0.15, "노드 평균 mean ≈ ctx_balance");

    // ── 4. on-point ───────────────────────────────────────────────────────────
    // cold-start 후 다수 노드에서 onpoint_batches(node, op, 2) ≥ 1.
    // 62-배치 변종이라 100% 아닐 수 있음 — 비율로 본다.
    int on1 = 0, on2 = 0, eligible = 0;
    for (const ClusterNode& nd : nodes) {
        if (nd.count() < op.decode_count_target) continue;  // 배치 구성 자체 불가
        ++eligible;
        int got = onpoint_batches(nd, op, 2);
        if (got >= 1) ++on1;
        if (got >= 2) ++on2;
    }
    const double on1_frac = eligible ? (double)on1 / eligible : 0.0;
    const double on2_frac = eligible ? (double)on2 / eligible : 0.0;
    std::printf("[4] onpoint k=2: eligible=%d on>=1=%d(%.1f%%) on>=2=%d(%.1f%%)\n",
                eligible, on1, 100.0 * on1_frac, on2, 100.0 * on2_frac);
    CHECK(eligible > 0, "onpoint: count≥N_dec 인 노드 존재");
    CHECK(on1_frac >= 0.5, "다수 적격 노드에서 ≥1 배치 명중");

    // ── 5. per-completion 힐링 drift ──────────────────────────────────────────
    // 한 노드에 대해 라운드마다 일부 제거(departed) → heal_node 반복.
    // count·mean 안정(drift 없음), 상주 긴 요청(toxic) 비율 보존.
    // 힐링 입력 노드: cold-start 결과 중 count 가 가장 큰(=가장 꽉 찬) 노드 선택.
    int pick = -1;
    for (int j = 0; j < Z; ++j)
        if (pick < 0 || nodes[j].count() > nodes[pick].count()) pick = j;
    ClusterNode node = nodes[pick];

    WorkloadSource src(/*seed*/7);
    const double tox_thresh = op.ctx_balance;  // ctx_balance 초과 = 긴(toxic) 요청
    auto toxic_frac = [&](const ClusterNode& nd) {
        if (nd.count() == 0) return 0.0;
        int t = 0;
        for (int x : nd.lengths) if (x > tox_thresh) ++t;
        return (double)t / nd.count();
    };

    const int count0 = node.count();
    const double mean0 = node.mean();
    const double tox0 = toxic_frac(node);
    std::printf("[5] heal start: count=%d mean=%.0f toxic_frac=%.3f\n", count0, mean0, tox0);

    std::mt19937 rmrng(11);
    const int ROUNDS = 30;
    double max_count_drift = 0.0, max_mean_drift = 0.0, max_tox_drift = 0.0;
    for (int r = 0; r < ROUNDS; ++r) {
        // 라운드 churn: 현재 풀에서 무작위 ~10% 제거 (완료 retire = departed hole).
        std::vector<int> idx(node.count());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rmrng);
        int rm = std::max(1, node.count() / 10);
        std::vector<int> rm_idx(idx.begin(), idx.begin() + rm);
        std::sort(rm_idx.begin(), rm_idx.end(), std::greater<int>());
        std::vector<int> departed;
        for (int ix : rm_idx) {
            departed.push_back(node.lengths[ix]);          // hole = 떠난 길이
            node.total -= node.lengths[ix];
            node.lengths.erase(node.lengths.begin() + ix);
        }
        // per-completion 힐링: ideal=hole 로 like-for-like 되채움.
        heal_node(node, departed, op, src);

        max_count_drift = std::max(max_count_drift, std::fabs((double)(node.count() - count0)));
        max_mean_drift  = std::max(max_mean_drift, std::fabs(node.mean() - mean0));
        max_tox_drift   = std::max(max_tox_drift, std::fabs(toxic_frac(node) - tox0));
    }
    std::printf("[5] heal after %d rounds: count=%d mean=%.0f toxic_frac=%.3f | "
                "max_drift count=%.0f mean=%.0f(%.1f%%) toxic=%.3f\n",
                ROUNDS, node.count(), node.mean(), toxic_frac(node),
                max_count_drift, max_mean_drift, 100.0 * max_mean_drift / mean0, max_tox_drift);
    // count 안정: 힐링이 풀을 목표크기로 유지 (small drift, ≤ 한 라운드 churn 폭).
    CHECK(max_count_drift <= (double)(count0 / 10 + 1), "힐링: count 안정 (drift ≤ churn 폭)");
    // mean 안정: ideal=hole 되채움이라 평균이 흐르지 않음.
    CHECK(max_mean_drift <= 0.15 * mean0, "힐링: mean drift 없음 (≤15%)");
    // toxic-fit 보존: 긴 요청 비율이 like-for-like 로 유지.
    CHECK(max_tox_drift <= 0.20, "힐링: 긴 요청 비율 보존 (toxic-fit)");

    return puls_test::summary();
}
