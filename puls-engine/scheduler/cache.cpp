// proto/scheduler — on-node KV 캐시 구현. CHECKLIST §3 cache.
// 근거: predictive_prepositioning_scheduler.md "on-node KV 캐시" 비용표
//       (Hit / Miss·SSD생존 / Miss·완전방출). 3-tier: HBM → SSD → Gone.
// 정책: 적격(len>threshold) ∧ 노드 잔여≥KV → HBM; 항상 SSD 기록(멀티턴 오프로드);
//       idle>evict_age → SSD 강등; idle>gone_age → 소멸. LRU·승격 없음.
#include "scheduler/cache.h"

namespace proto {

ClusterCache::ClusterCache(int num_nodes, long long node_capacity_bytes,
                           long long kv_bytes_per_token, const CacheConfig& cfg)
    : node_used_(num_nodes, 0),
      node_budget_(num_nodes, node_capacity_bytes),
      node_cap_(node_capacity_bytes),
      kv_bpt_(kv_bytes_per_token),
      cfg_(cfg) {}

// 완료: id 의 KV 를 회계. 기존 HBM 엔트리 있으면 먼저 바이트 회수 후 재배치.
void ClusterCache::on_complete(int node, long long id, int total_len, long long now) {
    const long long bytes = static_cast<long long>(total_len) * kv_bpt_;
    const bool eligible = total_len > cfg_.eligibility_threshold;

    // 기존 HBM 엔트리의 바이트 회수(재완료 = 멀티턴 누적 길이 갱신).
    auto it = idx_.find(id);
    if (it != idx_.end() && it->second.tier == Tier::HBM) {
        node_used_[it->second.node] -= static_cast<long long>(it->second.len) * kv_bpt_;
    }

    if (eligible && node_used_[node] + bytes <= node_budget_[node]) {
        idx_[id] = Entry{node, total_len, now, Tier::HBM};
        node_used_[node] += bytes;
    } else {
        // 적격 미만 또는 잔여 부족 → SSD 만(HBM 점유 0).
        idx_[id] = Entry{node, total_len, now, Tier::SSD};
    }
}

// 복귀 조회 = 비용 계층 판정. 조회 시 idle 리셋(.last = now).
ClusterCache::Lookup ClusterCache::lookup(long long id, long long now) {
    auto it = idx_.find(id);
    if (it == idx_.end()) {
        ++recomputes_;
        return Lookup{CacheResult::MissRecompute, -1, 0.0};
    }
    Entry& e = it->second;
    switch (e.tier) {
        case Tier::HBM:
            ++hits_;
            e.last = now;
            return Lookup{CacheResult::Hit, e.node, 0.0};
        case Tier::SSD: {
            ++reloads_;
            const double reload_cost =
                static_cast<double>(e.len) * kv_bpt_ / cfg_.offload_bw_bytes_per_round;
            e.last = now;
            return Lookup{CacheResult::MissReload, e.node, reload_cost};
        }
        case Tier::Gone:
        default:
            ++recomputes_;
            return Lookup{CacheResult::MissRecompute, -1, 0.0};
    }
}

// 부작용 없는 조회 — lookup 과 동일 판정, 카운터·idle 리셋 없음 (어피니티 라우팅 판단용).
ClusterCache::Lookup ClusterCache::peek(long long id) const {
    auto it = idx_.find(id);
    if (it == idx_.end()) return Lookup{CacheResult::MissRecompute, -1, 0.0};
    const Entry& e = it->second;
    switch (e.tier) {
        case Tier::HBM:
            return Lookup{CacheResult::Hit, e.node, 0.0};
        case Tier::SSD:
            return Lookup{CacheResult::MissReload, e.node,
                          static_cast<double>(e.len) * kv_bpt_ / cfg_.offload_bw_bytes_per_round};
        case Tier::Gone:
        default:
            return Lookup{CacheResult::MissRecompute, -1, 0.0};
    }
}

// 노드별 동적 예산 집행: used > budget 인 동안 그 노드 HBM 엔트리 중 idle 가장 오래된 것부터
// SSD 강등. on_complete admit 도 이 예산을 본다 (HBM 바이트 폐루프).
void ClusterCache::enforce_budget(int node, long long budget) {
    if (budget < 0) budget = 0;
    node_budget_[node] = budget;
    while (node_used_[node] > budget) {
        Entry* victim = nullptr;
        for (auto& kv : idx_) {
            Entry& e = kv.second;
            if (e.tier == Tier::HBM && e.node == node &&
                (victim == nullptr || e.last < victim->last)) {
                victim = &e;
            }
        }
        if (victim == nullptr) break;
        node_used_[node] -= static_cast<long long>(victim->len) * kv_bpt_;
        victim->tier = Tier::SSD;
    }
}

// 라운드 정리: HBM idle>evict_age → SSD 강등(바이트 회수); SSD idle>gone_age → 소멸.
void ClusterCache::evict_tick(long long now) {
    for (auto& kv : idx_) {
        Entry& e = kv.second;
        if (e.tier == Tier::HBM && now - e.last > cfg_.evict_age) {
            node_used_[e.node] -= static_cast<long long>(e.len) * kv_bpt_;
            e.tier = Tier::SSD;
        } else if (e.tier == Tier::SSD && now - e.last > cfg_.gone_age) {
            e.tier = Tier::Gone;  // 엔트리 유지 → lookup 이 recompute 보고.
        }
    }
}

}  // namespace proto
