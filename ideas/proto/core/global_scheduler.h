// PULS-ENGINE — ③ 글로벌(클러스터) 스케줄러. CONTRACT.md §3-③.
// 게이트(엣지 격리) + cold-start 분배 + per-node 힐링 + on-point 체크.
// inter-node swap 0, 무축출.
#pragma once
#include <vector>
#include "operating_point.h"
#include "request_source.h"

namespace puls {

// 한 노드의 상주 풀(길이 = KV footprint 근사). 글로벌 층 관점.
struct ClusterNode {
    std::vector<int> lengths;
    long long total = 0;
    int count() const { return (int)lengths.size(); }
    double mean() const { return lengths.empty() ? 0.0 : (double)total / lengths.size(); }
    void admit(int len) { lengths.push_back(len); total += len; }
};

// 게이트: 긴 것부터 shed 해 남은 평균 ≤ ctx_balance + edge_band.
// 반환 = {kept, edged}. (kept 는 cold-start 입력)
struct GateResult { std::vector<int> kept; std::vector<int> edged; };
GateResult gate(std::vector<int> draw, const OperatingPoint& op);

// cold-start: arrival순, min|추가후 mean − ctx_balance| 그리디. can_fit(cap ∧ count<node_max).
// nodes 를 in-place 채움. 반환 = leftover(배치 못한 수).
int cold_start(std::vector<int> kept, std::vector<ClusterNode>& nodes,
               const OperatingPoint& op, unsigned seed);

// per-node 힐링: 완료로 빠진 각 hole 을 ideal=hole 로 되채움(toxic-fit). batched 금지.
// 반환 = pull 수.
int heal_node(ClusterNode& node, const std::vector<int>& departed,
              const OperatingPoint& op, RequestSource& src);

// on-point 체크(검증용): disjoint k 개 (count, kv±band) 배치 compose. 명중 개수 반환.
int onpoint_batches(const ClusterNode& node, const OperatingPoint& op, int k);

} // namespace puls
