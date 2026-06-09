// proto/sim — 멀티턴 워크로드 구현. CHECKLIST §4 workload_mt. **sim 전용.**
// 분포 B(core/workload) + 엣지 게이팅 신규 도착, 결정론 max_tokens(EOS 무관),
// 길이의존 복귀확률(short<mid<long, 평균~2.5턴) + think-gap idle 재진입.
// 근거: ../predictive_prepositioning_scheduler.md ("동작" 완료=max_tokens,
//       "글로벌 풀 = 대용량 유한 큐" 복귀확률 길이의존·평균~2.5턴).
#include "sim/workload_mt.h"

#include "core/workload.h"

namespace proto {

MultiTurnWorkload::MultiTurnWorkload(unsigned seed, int edge_cutoff,
                                     const MultiTurnConfig& cfg)
    : rng_(seed), edge_cutoff_(edge_cutoff), cfg_(cfg) {}

// 분포 B 표본 중 interior(≤ edge_cutoff)만 통과. 컷오프 초과 long 은 엣지 노드行.
int MultiTurnWorkload::gated_sample() {
    int v;
    do {
        v = puls::sample_distribution_b(rng_);
    } while (v > edge_cutoff_);
    return v;
}

// prompt 길이 → class: 0 short(<16000) / 1 mid(<256000) / 2 long(else). (분포 B 경계)
static int cls(int prompt) {
    if (prompt < 16000)  return 0;
    if (prompt < 256000) return 1;
    return 2;
}

Request MultiTurnWorkload::fresh_arrival(long long now) {
    std::uniform_int_distribution<int> dtot(cfg_.dtot_lo, cfg_.dtot_hi);
    Request r;
    r.id            = next_id_++;
    r.prompt        = gated_sample();
    r.max_tokens    = dtot(rng_);
    r.cached_prefix = 0;
    r.arrival       = now;
    r.turn          = 0;
    return r;
}

MultiTurnWorkload::Return MultiTurnWorkload::maybe_return(const Request& prev,
                                                         long long now) {
    double p;
    switch (cls(prev.prompt)) {
        case 0:  p = cfg_.p_return_short; break;
        case 1:  p = cfg_.p_return_mid;   break;
        default: p = cfg_.p_return_long;  break;
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    bool returns = (uni(rng_) < p);
    if (!returns) return Return{false, 0, Request{}};

    std::uniform_int_distribution<int> think(cfg_.think_gap_lo, cfg_.think_gap_hi);
    std::uniform_int_distribution<int> newmsg(cfg_.new_msg_lo, cfg_.new_msg_hi);
    std::uniform_int_distribution<int> dtot(cfg_.dtot_lo, cfg_.dtot_hi);

    int think_gap = think(rng_);
    long long return_round = now + think_gap;

    Request next;
    next.id            = prev.id;
    next.turn          = prev.turn + 1;
    next.cached_prefix = prev.prompt + prev.max_tokens; // P1 = 이전 컨텍스트 + 생성분
    int p2             = newmsg(rng_);                  // 새 메시지 P2
    next.prompt        = next.cached_prefix + p2;       // P1 + P2
    next.max_tokens    = dtot(rng_);
    next.arrival       = return_round;
    return Return{true, return_round, next};
}

} // namespace proto
