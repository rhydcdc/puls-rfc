// proto/scheduler — on-node KV 캐시(노드별 HBM 용량) + 글로벌 id 인덱스(3-tier).
// CHECKLIST §3 cache. capacity = derive 잔여(하드코딩 0). admit = 잔여 ≥ 크기. evict = idle age.
#pragma once
#include <vector>
#include <unordered_map>
#include "scheduler/request.h"

namespace proto {

// 복귀 조회 결과 = 비용 계층. (idea 문서 비용표)
//   Hit          : HBM 상주 → 재로드 0, P1 재계산 0.
//   MissReload   : HBM 방출됐으나 SSD 생존 → 재로드 = 크기/속도, P1 재계산 0.
//   MissRecompute: 완전 소멸 → P1 전체 재계산(드라이버가 prefill 비용으로 지불).
enum class CacheResult { Hit, MissReload, MissRecompute };

struct CacheConfig {
    int    eligibility_threshold;       // HBM 캐싱 적격 최소 prompt 길이 (스윕)
    int    evict_age;                   // HBM idle 라운드 상한 → SSD 강등 (스윕)
    int    gone_age;                    // SSD idle 라운드 상한 → 완전 소멸 (고정 가정, 비-스윕)
    double offload_bw_bytes_per_round;  // 재로드 속도 (바이트/라운드, 가정 라벨)
};

// 클러스터 캐시: 노드별 HBM 용량 회계 + id→엔트리 글로벌 인덱스.
class ClusterCache {
public:
    ClusterCache(int num_nodes, long long node_capacity_bytes,
                 long long kv_bytes_per_token, const CacheConfig& cfg);

    // 완료: node 에서 id(전체 길이 total_len tokens) 디코드 완료.
    //   적격(total_len > eligibility_threshold) ∧ 노드 HBM 잔여 ≥ KV크기 → HBM 캐싱.
    //   항상 SSD 기록(멀티턴 오프로드 전제). 적격 미만은 HBM 미캐싱(SSD 만).
    void on_complete(int node, long long id, int total_len, long long now);

    struct Lookup { CacheResult result; int node; double reload_cost; };
    // 복귀 조회. Hit → reload 0, node = 보유 노드(히트 라우팅 힌트). 조회 시 idle 리셋.
    // 미기록(첫 턴 등) id → MissRecompute(node = -1).
    Lookup lookup(long long id, long long now);

    // 부작용 없는 조회(카운터·idle 리셋 없음) — 어피니티 라우팅 판단용(csched ①-순위).
    Lookup peek(long long id) const;

    // 라운드마다: HBM idle > evict_age → SSD 강등; SSD idle > gone_age → 소멸.
    void evict_tick(long long now);

    // 노드별 동적 예산(HBM 바이트 폐루프): 멀티턴 인플레이션으로 풀 실 KV 가 설계 footprint 를
    // 초과한 만큼 캐시 예산을 깎는다. budget 미만이 될 때까지 그 노드 HBM 엔트리를 idle 오래된
    // 순으로 SSD 강등. 이후 on_complete admit 도 이 예산으로 판정.
    void enforce_budget(int node, long long budget);

    long long node_used(int node) const { return node_used_[node]; }
    long long node_capacity() const { return node_cap_; }
    long long hbm_hits() const { return hits_; }
    long long ssd_reloads() const { return reloads_; }
    long long recomputes() const { return recomputes_; }

private:
    enum class Tier { HBM, SSD, Gone };
    struct Entry { int node; int len; long long last; Tier tier; };
    std::unordered_map<long long, Entry> idx_;
    std::vector<long long> node_used_;   // 노드별 HBM 사용 바이트
    std::vector<long long> node_budget_; // 노드별 동적 예산 (기본 = node_cap_)
    long long node_cap_;
    long long kv_bpt_;                   // kv_bytes_per_token
    CacheConfig cfg_;
    long long hits_ = 0, reloads_ = 0, recomputes_ = 0;
};

} // namespace proto
