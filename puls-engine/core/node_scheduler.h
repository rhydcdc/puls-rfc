// PULS-ENGINE — ② 노드 스케줄러. CONTRACT.md §3-②.
// 한 노드의 decode/prefill 풀 관리 + 센터링 admit + per-completion 힐링.
// 새 요청은 RequestSource 로만 얻는다(runtime/sim 공유).
#pragma once
#include <vector>
#include "operating_point.h"
#include "steering.h"
#include "request_source.h"

namespace puls {

// 한 노드의 상주 디코드 풀(길이-도메인). live KV = prompt + 누적 decode.
struct NodeDecoder {
    int prompt;
    int dtot;     // 총 decode 길이
    int dec;      // 누적 decode 진행
    int wait;
    long long live_kv() const { return (long long)prompt + dec; }
};

class NodeScheduler {
public:
    NodeScheduler(const OperatingPoint& op, RequestSource& src);

    // 센터링 admit: ideal = ctx_balance×(cnt+1) − liveSum. 새 디코더는 dec=0 으로 admit.
    void admit_centered();

    // 디코드 한 μ-batch 구성(steering+age-cap). used 공유로 2회 호출 → disjoint 246 advance.
    // 반환 = 선택 인덱스. 진행/wait 갱신은 advance_round 가 일괄 수행.
    std::vector<int> compose_decode_batch(std::vector<char>& used) const;

    // 한 라운드: 2 μ-batch advance(dec++/wait++) → 완료 retire → per-completion 힐링으로
    // 풀을 decode_pool 로 복구(ideal=hole). 완료 수 반환.
    int advance_round();

    const std::vector<NodeDecoder>& pool() const { return pool_; }
    double mean_live_kv() const;
    long long sum_live_kv() const;

private:
    OperatingPoint op_;
    RequestSource& src_;
    std::vector<NodeDecoder> pool_;
};

} // namespace puls
