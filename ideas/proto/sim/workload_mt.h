// proto/sim — 멀티턴 워크로드 (sim 전용). CHECKLIST §4 workload_mt.
// 분포B+게이팅 신규 도착, 결정론 max_tokens, 길이의존 복귀확률 + think-gap idle.
#pragma once
#include <random>
#include "scheduler/request.h"

namespace proto {

struct MultiTurnConfig {
    double p_return_short;            // class별 복귀확률 (short < mid < long), 평균 ~2.5턴
    double p_return_mid;
    double p_return_long;
    int    dtot_lo, dtot_hi;          // max_tokens = uniform[lo,hi] (결정론 decode 길이)
    int    new_msg_lo, new_msg_hi;    // 복귀 시 새 메시지 P2 = uniform[lo,hi]
    int    think_gap_lo, think_gap_hi;// 복귀까지 idle 라운드 (캐시 evict-age 와 대조)
};

// 멀티턴 워크로드 생성기. (sim 전용 — 분포 B + 확률 churn, CONTRACT §2 경계)
class MultiTurnWorkload {
public:
    MultiTurnWorkload(unsigned seed, int edge_cutoff, const MultiTurnConfig& cfg);

    // 신규 도착 1건: 분포 B + 엣지 게이팅, 첫 턴(turn=0), 새 id, cached_prefix=0.
    Request fresh_arrival(long long now);

    struct Return { bool returns; long long return_round; Request next; };
    // prev 완료 후 복귀 여부 + 복귀 라운드(now + think_gap) + 다음 턴 요청.
    //   다음 턴: 같은 id, turn+1, cached_prefix = prev.prompt + prev.max_tokens (= P1),
    //            prompt = P1 + 새 메시지 P2, max_tokens 재샘플. returns=false 면 next 무의미.
    Return maybe_return(const Request& prev, long long now);

private:
    std::mt19937    rng_;
    int             edge_cutoff_;
    MultiTurnConfig cfg_;
    long long       next_id_ = 0;
    int             gated_sample();   // 분포 B 표본 중 ≤ edge_cutoff 만 (interior)
};

} // namespace proto
