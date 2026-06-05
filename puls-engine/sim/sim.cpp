// PULS-ENGINE — 시뮬레이터 드라이버. CONTRACT.md §2 (sim = WorkloadSource / 분포 B) / §3-③.
//
// 경계: sim 은 분포 B(WorkloadSource = 무한풀 emulation) 전용이다 — runtime 과 갈리는 단일 지점이
//       RequestSource 다. 여기서는 분포 B 로 draw 를 만들고(gate 입력), cold-start 로 노드에 분배한
//       뒤, 여러 라운드 churn + per-completion 힐링을 돌려 동작점 명중·Σdev·count 범위·drift 를
//       인쇄한다. 동작점은 derive 로 산출한 op 로 구동(하드코딩 0).
//
// 근거 의도: implementation/analysis/cluster_balance.cpp 의 sim1(cold-start) / sim3(churn+힐링).
//            거기 손으로 박았던 256/123/12.3M/100K 를 derive 산출 op 로 대체해 core 로 재현한다.
//            분포 B / 무한풀 / 확률 churn 은 sim 전용(CONTRACT §4 · §9-2).

#include "core/derive.h"
#include "core/global_scheduler.h"
#include "core/operating_point.h"
#include "core/workload.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <random>
#include <vector>

using namespace puls;

namespace {

// 측정: count 범위 적중률, |dev|(mean−ctx) 평균/최대, on-point(2 disjoint 배치) 적중 노드%.
struct Stat {
    int    cnt_min = INT_MAX;
    double cnt_mean = 0;
    double in_range = 0;   // count ∈ [node_min, node_max] 노드 %
    double dev_avg = 0;    // |mean − ctx_balance| 평균
    double dev_max = 0;
    double on2 = 0;        // 2 disjoint 배치 명중 노드 %
};

Stat measure(const std::vector<ClusterNode>& nodes, const OperatingPoint& op) {
    Stat s;
    int in = 0, o2 = 0;
    double csum = 0, dsum = 0, dmax = 0;
    const int Z = static_cast<int>(nodes.size());
    for (const auto& nd : nodes) {
        int c = nd.count();
        s.cnt_min = std::min(s.cnt_min, c);
        csum += c;
        if (c >= op.node_min && c <= op.node_max) ++in;
        double dv = std::fabs(nd.mean() - op.ctx_balance);
        dsum += dv;
        dmax = std::max(dmax, dv);
        if (onpoint_batches(nd, op, 2) >= 2) ++o2;
    }
    s.cnt_mean = csum / Z;
    s.in_range = 100.0 * in / Z;
    s.dev_avg = dsum / Z;
    s.dev_max = dmax;
    s.on2 = 100.0 * o2 / Z;
    return s;
}

}  // namespace

int main() {
    // ── 동작점: derive 로 산출(하드코딩 0) ───────────────────────────────────────
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6, /*gpus_a*/8, /*gpus_b*/8};
    const OperatingPoint op = derive_operating_point(llama, b200, /*prefill*/128);

    const int Z = 256;                 // 클러스터 노드 수(sim 규모 — 동작점과 무관한 sim knob)
    const unsigned SEED = 12345;

    std::printf("===== PULS 시뮬레이터 (WorkloadSource — 분포 B, 무한풀 emulation) =====\n");
    std::printf("동작점(derive: Llama70B+B200+prefill128): ctx=%.0f N_dec=%d kv=%lld "
                "node_min=%d node_max=%d decode_pool=%d\n",
                op.ctx_balance, op.decode_count_target, op.kv_operating_target,
                op.node_min, op.node_max, op.decode_pool);
    std::printf("분포 B: short20%%[1-16K]/mid70%%[16-256K]/long10%%[256K-1M]. Z=%d nodes. "
                "on2 = 2 disjoint (N_dec, kv±band) 배치 명중 노드%%.\n\n", Z);

    // ── sim1: cold-start (gate → cold_start) — edge_band 스윕 ──────────────────────
    // 분포 B 로 P개 draw → gate(긴 것 shed) → cold_start(arrival순 min|mean−ctx| greedy).
    std::printf("----- sim1: cold-start (gate edge%% = f(E)) — edge_band 스윕 -----\n");
    std::printf("%7s | %6s %7s | %7s %8s | %8s %8s | %6s\n",
                "E(K)", "edge#", "edge%", "cntMin", "in-range", "|dev|avg", "|dev|max", "on2%");
    const int P_DRAW = Z * op.node_max;   // 노드를 채울 만큼 충분히 draw
    std::vector<double> Es = {0, 1000, 2000, 5000, 10000};
    for (double E : Es) {
        OperatingPoint ope = op;
        ope.edge_band = E;
        // 분포 B draw (sim 전용 RNG).
        std::mt19937 draw_rng(SEED);
        std::vector<int> draw(P_DRAW);
        for (auto& x : draw) x = sample_distribution_b(draw_rng);

        GateResult g = gate(draw, ope);
        std::vector<ClusterNode> nodes(Z);
        int leftover = cold_start(g.kept, nodes, ope, SEED);

        Stat s = measure(nodes, ope);
        int edge = static_cast<int>(g.edged.size()) + leftover;
        std::printf("%7.0f | %6d %7.2f | %7d %7.1f%% | %8.0f %8.0f | %5.1f\n",
                    E / 1000, edge, 100.0 * edge / P_DRAW, s.cnt_min, s.in_range,
                    s.dev_avg, s.dev_max, s.on2);
    }

    // ── sim2: steady-state churn + per-completion 힐링 (drift = early vs late) ──────
    // E=1K cold-start 후, 매 라운드 각 상주가 완료확률 p 로 retire → 각 완료 hole 을 WorkloadSource
    // 에서 ideal=hole 로 보충(heal_node, toxic-fit). early(warmup 직후) vs late(말기 평균) 으로
    // drift 없음(early≈late)을 확인한다.
    std::printf("\n----- sim2: churn + per-completion 힐링 (E=1K cold-start, p 완료확률) -----\n");
    std::printf("%5s %7s | %7s %8s | %8s %8s | %6s %9s\n",
                "p%", "window", "cntMin", "in-range", "|dev|avg", "|dev|max", "on2%", "pulls/rd");
    const int ROUNDS = 300, WARM = 150;
    std::vector<double> Ps = {0.01, 0.03, 0.05};
    for (double p : Ps) {
        OperatingPoint op1k = op;
        op1k.edge_band = 1000.0;

        std::mt19937 draw_rng(SEED);
        std::vector<int> draw(P_DRAW);
        for (auto& x : draw) x = sample_distribution_b(draw_rng);
        GateResult g = gate(draw, op1k);
        std::vector<ClusterNode> nodes(Z);
        cold_start(g.kept, nodes, op1k, SEED);

        WorkloadSource src(SEED + 7);     // 힐링용 분포 B 출처(무한풀 emulation)
        std::mt19937 churn_rng(SEED + 99); // 완료 churn RNG (sim 전용)
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        Stat early{};
        double l_in = 0, l_da = 0, l_dm = 0, l_o2 = 0, l_cm = 0, l_pull = 0;
        int l_cmin = INT_MAX, l_n = 0;
        for (int rd = 0; rd < ROUNDS; ++rd) {
            long long pulls = 0;
            for (auto& nd : nodes) {
                // 완료확률 p 로 retire — departed(완료 길이들) 수집, 나머지 유지.
                std::vector<int> keep, departed;
                long long s = 0;
                for (int L : nd.lengths) {
                    if (uni(churn_rng) >= p) { keep.push_back(L); s += L; }
                    else departed.push_back(L);
                }
                nd.lengths.swap(keep);
                nd.total = s;
                // per-completion 힐링: 각 hole 을 ideal=hole 로 like-for-like 보충(core).
                pulls += heal_node(nd, departed, op1k, src);
            }
            if (rd == WARM) early = measure(nodes, op1k);
            if (rd >= WARM) {
                Stat st = measure(nodes, op1k);
                l_in += st.in_range; l_da += st.dev_avg; l_dm = std::max(l_dm, st.dev_max);
                l_o2 += st.on2; l_cmin = std::min(l_cmin, st.cnt_min);
                l_cm += st.cnt_mean; l_pull += pulls; ++l_n;
            }
        }
        std::printf("%5.0f %7s | %7d %7.1f%% | %8.0f %8.0f | %5.1f %9s\n",
                    p * 100, "early", early.cnt_min, early.in_range, early.dev_avg,
                    early.dev_max, early.on2, "-");
        std::printf("%5s %7s | %7d %7.1f%% | %8.0f %8.0f | %5.1f %9.0f\n",
                    "", "late", l_cmin, l_in / l_n, l_da / l_n, l_dm, l_o2 / l_n, l_pull / l_n);
    }
    std::printf("(early=warmup 직후 1라운드, late=말기 %d라운드 평균. drift 없으면 early≈late.)\n",
                ROUNDS - WARM);
    std::printf("\n→ sim 경로: 분포 B(WorkloadSource)로만 요청 공급 — gate→cold_start→churn→힐링이\n"
                "  전부 core 로직. runtime(실 큐)과 코드 공유, RequestSource 한 지점에서 갈림.\n");
    return 0;
}
