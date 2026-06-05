// PULS-ENGINE — ③ 글로벌(클러스터) 스케줄러 구현. CONTRACT.md §3-③ / §4.
// 근거: cluster_routing.py(gate/cold_start/heal/disjoint_onpoint_batches)
//       + cluster_balance.cpp(gate/place_greedy/onpoint_k). canonical(§4): per-completion 힐링
//       (ideal=hole, batched 금지). 동작점 상수는 전부 op 에서 — 리터럴 0(§9-5).
//   cap = node_max × ctx_balance (노드 footprint 캡 = NODE_MAX×ctx, 레퍼런스의 30M 등가).
#include "core/global_scheduler.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace puls {

// ── 게이트(엣지 격리) ─────────────────────────────────────────────────────────
// draw 내림차순 정렬 후, 남은 평균 ≤ ctx_balance + edge_band 가 될 때까지 맨 앞(최장) shed.
// 반환 {kept=draw[i:], edged=draw[:i]}.
GateResult gate(std::vector<int> draw, const OperatingPoint& op) {
    std::sort(draw.begin(), draw.end(), std::greater<int>());
    long long total = 0;
    for (int x : draw) total += x;
    int n = static_cast<int>(draw.size());
    int i = 0;
    const double ceil = op.ctx_balance + op.edge_band;
    while (n > 0 && total / static_cast<double>(n) > ceil) {
        total -= draw[i];
        ++i;
        --n;
    }
    GateResult r;
    r.kept.assign(draw.begin() + i, draw.end());
    r.edged.assign(draw.begin(), draw.begin() + i);
    return r;
}

// ── cold-start 분배 ───────────────────────────────────────────────────────────
// arrival순(seed shuffle), 각 길이를 min|추가후 mean − ctx_balance| 노드에 그리디 적재.
// can_fit = (nd.total+len ≤ cap) ∧ (nd.count() < node_max). cap = node_max×ctx_balance.
// nodes 는 호출자가 크기 Z 로 미리 생성. 못 배치한 수(leftover) 반환.
int cold_start(std::vector<int> kept, std::vector<ClusterNode>& nodes,
               const OperatingPoint& op, unsigned seed) {
    std::mt19937 g(seed);
    std::shuffle(kept.begin(), kept.end(), g);
    const long long cap =
        static_cast<long long>(op.node_max) * static_cast<long long>(op.ctx_balance);
    int leftover = 0;
    for (int len : kept) {
        int best = -1;
        double best_dev = 1e18;
        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            const ClusterNode& nd = nodes[j];
            if (nd.total + len > cap || nd.count() >= op.node_max) continue;
            const double dev =
                std::fabs((nd.total + len) / static_cast<double>(nd.count() + 1) - op.ctx_balance);
            if (dev < best_dev) {
                best_dev = dev;
                best = j;
            }
        }
        if (best < 0) {
            ++leftover;
            continue;
        }
        nodes[best].admit(len);
    }
    return leftover;
}

// ── per-node 힐링 (per-completion, toxic-fit) ─────────────────────────────────
// 각 hole(departed)에 대해 ideal=hole 로 like-for-like 되채움. batched(평균) 금지(§4).
// cap_room ≤ 0 이면 중단. src.pull_near 가 prompt≥0 이면 admit. pull 수 반환.
int heal_node(ClusterNode& node, const std::vector<int>& departed,
              const OperatingPoint& op, RequestSource& src) {
    const long long cap =
        static_cast<long long>(op.node_max) * static_cast<long long>(op.ctx_balance);
    int pulls = 0;
    for (int hole : departed) {
        const long long cap_room = cap - node.total;
        if (cap_room <= 0) break;
        PulledRequest req = src.pull_near(static_cast<double>(hole), cap_room);  // ideal = hole
        if (req.prompt < 0) break;
        node.admit(req.prompt);
        ++pulls;
    }
    return pulls;
}

// ── on-point 체크(검증용) ─────────────────────────────────────────────────────
// node.lengths 정렬 후 disjoint k 개 (decode_count_target, kv_operating_target±idle_band) 배치를
// closest-to-ideal 그리디로 compose. 실패 배치는 used 롤백·중단. 명중 개수 반환.
int onpoint_batches(const ClusterNode& node, const OperatingPoint& op, int k) {
    const int m = node.count();
    if (m < op.decode_count_target) return 0;
    std::vector<long long> a(node.lengths.begin(), node.lengths.end());
    std::sort(a.begin(), a.end());
    std::vector<char> used(m, 0);
    int got = 0;
    for (int b = 0; b < k; ++b) {
        long long S = 0;
        int n = 0;
        std::vector<int> pick;
        while (n < op.decode_count_target && S < op.kv_operating_target) {
            const long long ideal = std::llround(
                static_cast<double>(op.kv_operating_target - S) / (op.decode_count_target - n));
            int pos = static_cast<int>(std::lower_bound(a.begin(), a.end(), ideal) - a.begin());
            int Lp = pos - 1, Rp = pos;
            while (Lp >= 0 && used[Lp]) --Lp;
            while (Rp < m && used[Rp]) ++Rp;
            int best;
            if (Lp < 0 && Rp >= m) break;
            else if (Lp < 0) best = Rp;
            else if (Rp >= m) best = Lp;
            else best = ((ideal - a[Lp]) <= (a[Rp] - ideal)) ? Lp : Rp;
            used[best] = 1;
            pick.push_back(best);
            S += a[best];
            ++n;
        }
        if (n == op.decode_count_target &&
            std::fabs(static_cast<double>(S) - static_cast<double>(op.kv_operating_target)) /
                    static_cast<double>(op.kv_operating_target) <= op.idle_band) {
            ++got;
        } else {
            for (int idx : pick) used[idx] = 0;
            break;
        }
    }
    return got;
}

} // namespace puls
