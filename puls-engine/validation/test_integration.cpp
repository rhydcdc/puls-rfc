// 검증 — 통합 한 줄기: derive → global(gate+cold-start) → node(per-node 힐링). CONTRACT.md §7 통합.
// 한 줄기:
//   ① op=derive(Llama70B, B200, 128).  WorkloadSource src(분포 B 무한풀 emulation).
//   ② 분포 B draw → gate(엣지 격리) → cold_start(nodes Z=64) → 일부 노드 heal_node 라운드 반복.
// 검증 포인트:
//   - 동작점 명중: cold-start 후 노드들에서 on-point(count, kv±band) 배치가 다수 compose(onpoint≥2).
//   - cold-start 후 노드 count ∈ [node_min, node_max] 가 다수.
//   - healing 후 drift 없음: 힐링 라운드 전후 노드 mean 이 ctx_balance 근방 유지(early≈late).
//   - inter-node swap 0: heal_node 는 src(분포 B)에서만 당긴다 — 구조적 CHECK(타 노드 합 불변).
//   - 다른 모델/HW(예: gpus_a=4) 로도 derive→스케줄 파이프가 자체정합(명중)인지 1 케이스.
#include "test_framework.h"
#include "core/derive.h"
#include "core/global_scheduler.h"
#include "core/workload.h"
#include <cstdio>
#include <vector>
#include <random>

using namespace puls;

// 분포 B 로 draw 개를 뽑아 길이 벡터 생성(검증 입력 트래픽).
static std::vector<int> draw_b(std::mt19937& rng, int n) {
    std::vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) v.push_back(sample_distribution_b(rng));
    return v;
}

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6, /*gpus_a*/8, /*gpus_b*/8};

    // ── ① derive ──────────────────────────────────────────────────────────────
    OperatingPoint op = derive_operating_point(llama, b200, 128);
    std::printf("derive: ctx=%.0f N_dec=%d kv=%lld node_min=%d node_max=%d edge=%.0f\n",
                op.ctx_balance, op.decode_count_target, op.kv_operating_target,
                op.node_min, op.node_max, op.edge_band);
    CHECK(op.decode_count_target > 0 && op.ctx_balance > 0, "derive produced a usable op-point");

    // ── ② 분포 B draw → gate ───────────────────────────────────────────────────
    std::mt19937 rng(12345);
    // 노드당 ~node_min 정도를 채울 만큼 충분히 draw (Z=64 노드).
    const int Z = 64;
    std::vector<int> draw = draw_b(rng, Z * op.node_min + Z * 5);
    GateResult g = gate(draw, op);
    std::printf("gate: draw=%zu kept=%zu edged=%zu\n",
                draw.size(), g.kept.size(), g.edged.size());
    // 게이트 후 남은 평균은 ctx_balance + edge_band 이하여야 한다(엣지 격리 불변식).
    {
        long long s = 0;
        for (int x : g.kept) s += x;
        double mean = g.kept.empty() ? 0.0 : (double)s / g.kept.size();
        CHECK(mean <= op.ctx_balance + op.edge_band + 1.0, "gate: kept mean <= ctx_balance + edge_band");
    }

    // ── ② cold-start (nodes Z=64) ──────────────────────────────────────────────
    std::vector<ClusterNode> nodes(Z);
    int leftover = cold_start(g.kept, nodes, op, /*seed*/777);
    std::printf("cold_start: leftover=%d\n", leftover);

    // cold-start 후 노드 count ∈ [node_min, node_max] 다수 검증.
    int in_band = 0, populated = 0;
    int onpoint_hits = 0;
    double sum_mean = 0.0; int mean_cnt = 0;
    for (const auto& nd : nodes) {
        if (nd.count() == 0) continue;
        ++populated;
        if (nd.count() >= op.node_min && nd.count() <= op.node_max) ++in_band;
        // 노드별 mean 은 cold-start 그리디가 ctx_balance 로 센터링.
        sum_mean += nd.mean();
        ++mean_cnt;
        // on-point: 이 노드에서 disjoint 1 배치 명중 가능?
        if (onpoint_batches(nd, op, 1) >= 1) ++onpoint_hits;
    }
    double avg_node_mean = mean_cnt ? sum_mean / mean_cnt : 0.0;
    std::printf("cold_start: populated=%d in_band[node_min,node_max]=%d onpoint_hits=%d avg_node_mean=%.0f\n",
                populated, in_band, onpoint_hits, avg_node_mean);
    CHECK(populated > 0, "cold_start populated some nodes");
    // 다수(과반)의 채워진 노드가 [node_min, node_max] 안.
    CHECK(in_band * 2 >= populated, "cold_start: majority of nodes count in [node_min,node_max]");
    // 동작점 명중: 다수 노드가 on-point 배치 한 개 이상 compose(on2 비율 대용 = 다수 명중).
    CHECK(onpoint_hits * 2 >= populated, "cold_start: majority of nodes hit an on-point batch");
    // 센터링: 노드 평균이 ctx_balance 근방.
    CHECK_REL(avg_node_mean, op.ctx_balance, 0.30, "cold_start: node means centered near ctx_balance");

    // ── ② per-node 힐링 라운드 반복: drift 없음 + inter-node swap 0 ──────────────
    // 첫 번째 채워진 노드를 골라 완료(departed)→heal_node 를 여러 라운드 반복.
    WorkloadSource src(/*seed*/2024);
    int target = -1;
    for (int j = 0; j < Z; ++j) {
        if (nodes[j].count() >= op.node_min) { target = j; break; }
    }
    CHECK(target >= 0, "found a target node to heal");

    if (target >= 0) {
        // inter-node swap 0 구조 검증: 다른 노드들의 합 스냅샷.
        std::vector<long long> other_totals_before(Z, 0);
        for (int j = 0; j < Z; ++j) other_totals_before[j] = nodes[j].total;

        double mean_early = nodes[target].mean();

        // 라운드마다: 몇 개를 완료(departed)로 빼고(짧은 것부터 — 완료 순간 churn) heal_node 로 되채움.
        const int rounds = 20;
        std::mt19937 pick_rng(55);
        for (int r = 0; r < rounds; ++r) {
            ClusterNode& nd = nodes[target];
            int c = nd.count();
            if (c == 0) break;
            // 완료 수 = 풀의 ~10% (per-completion 힐링 단위).
            int ndep = c / 10; if (ndep < 1) ndep = 1;
            // 임의의 hole 들을 제거(완료) — departed 길이 수집, 노드에서 제거.
            std::vector<int> departed;
            for (int d = 0; d < ndep && !nd.lengths.empty(); ++d) {
                std::uniform_int_distribution<int> di(0, (int)nd.lengths.size() - 1);
                int idx = di(pick_rng);
                int len = nd.lengths[idx];
                departed.push_back(len);
                nd.total -= len;
                nd.lengths.erase(nd.lengths.begin() + idx);
            }
            // per-completion 힐링: 각 hole 을 ideal=hole 로 like-for-like 되채움.
            int pulls = heal_node(nd, departed, op, src);
            (void)pulls;
        }

        double mean_late = nodes[target].mean();
        std::printf("heal: target=%d mean_early=%.0f mean_late=%.0f\n",
                    target, mean_early, mean_late);
        // drift 없음: 힐링 전후 노드 mean 이 ctx_balance 근방에서 안정(early≈late, 둘 다 근방).
        CHECK_REL(mean_late, mean_early, 0.25, "healing: no drift (early ~= late)");
        CHECK_REL(mean_late, op.ctx_balance, 0.40, "healing: mean stays near ctx_balance");

        // inter-node swap 0: heal_node 는 src(분포 B)에서만 당긴다 → 타 노드 합 불변.
        int other_changed = 0;
        for (int j = 0; j < Z; ++j) {
            if (j == target) continue;
            if (nodes[j].total != other_totals_before[j]) ++other_changed;
        }
        CHECK(other_changed == 0, "inter-node swap 0: healing touched no other node");
    }

    // ── 추가: 다른 HW(gpus_a=4)로 derive→스케줄 파이프 자체정합(명중) 1 케이스 ──────
    {
        HwSpec b200_4{/*tflops*/2200.0, /*mfu*/0.6, /*gpus_a*/4, /*gpus_b*/4};
        OperatingPoint op4 = derive_operating_point(llama, b200_4, 128);
        std::printf("alt-HW(gpus=4): ctx=%.0f N_dec=%d kv=%lld node_min=%d\n",
                    op4.ctx_balance, op4.decode_count_target, op4.kv_operating_target, op4.node_min);
        CHECK(op4.decode_count_target > 0 && op4.ctx_balance > 0, "alt-HW derive usable");

        std::mt19937 rng4(99);
        std::vector<int> draw4 = draw_b(rng4, Z * op4.node_min + Z * 5);
        GateResult g4 = gate(draw4, op4);
        std::vector<ClusterNode> nodes4(Z);
        cold_start(g4.kept, nodes4, op4, /*seed*/321);

        // 자체정합: 그 HW 의 동작점으로 그 HW 의 노드들이 on-point 명중하는지 다수.
        int pop4 = 0, hit4 = 0;
        for (const auto& nd : nodes4) {
            if (nd.count() == 0) continue;
            ++pop4;
            if (onpoint_batches(nd, op4, 1) >= 1) ++hit4;
        }
        std::printf("alt-HW(gpus=4): populated=%d onpoint_hits=%d\n", pop4, hit4);
        CHECK(pop4 > 0, "alt-HW cold_start populated nodes");
        CHECK(hit4 * 2 >= pop4, "alt-HW: self-consistent on-point hit (majority)");
    }

    return puls_test::summary();
}
