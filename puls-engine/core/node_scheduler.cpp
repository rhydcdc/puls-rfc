// PULS-ENGINE — ② 노드 스케줄러 구현. CONTRACT.md §3-② / §4 (canonical).
// 근거: cluster_lifecycle.cpp (admitDecodeCentered / composeDecodeBatch 2회 used 공유 /
//       main 루프의 advance(dec++/wait++)·retire·per-completion 힐링·풀 유지).
// canonical(§4·§9): 힐링은 per-completion(ideal=hole) 만 — batched(평균) 힐링 코드 부재.
//                   하나의 hole 당 정확히 하나 admit (개수=완료수, 크기=hole).
#include "core/node_scheduler.h"

#include <climits>
#include <vector>

namespace puls {

NodeScheduler::NodeScheduler(const OperatingPoint& op, RequestSource& src)
    : op_(op), src_(src) {}

// 노드 footprint 캡이 op 에 없으므로 충분히 큰 cap_room 을 둔다(노드 KV 캡 ≈ 무한).
// LLONG_MAX 직접 쓰면 합산 오버플로 위험이 있어 동작점 스케일의 넉넉한 배수로 둔다.
static long long node_cap_room(const OperatingPoint& op) {
    return static_cast<long long>(op.decode_pool) *
           static_cast<long long>(op.ctx_balance) * 4;
}

// 센터링 admit: ideal = ctx_balance×(cnt+1) − liveSum 으로 풀 live KV 평균을 ctx_balance 에 센터.
// 풀이 떠 있으면 ideal↓ → 작은 것을 당겨 평균을 끌어내린다. 새 디코더는 dec=0 으로 admit
// (콜드스타트의 랜덤 진행이 필요하면 sim 드라이버가 별도 주입 — NodeScheduler 는 RNG 없음).
void NodeScheduler::admit_centered() {
    long long live_sum = sum_live_kv();
    int cnt = static_cast<int>(pool_.size());
    double ideal = op_.ctx_balance * (cnt + 1) - static_cast<double>(live_sum);
    if (ideal < 1000) ideal = 1000;
    PulledRequest r = src_.pull_near(ideal, node_cap_room(op_));
    if (r.prompt >= 0) {
        NodeDecoder q;
        q.prompt = r.prompt;
        q.dtot = r.dtot;
        q.dec = 0;  // 갓 admit — 진행 0 (콜드스타트 랜덤 진행은 드라이버 주입).
        q.wait = 0;
        pool_.push_back(q);
    }
}

// 디코드 한 μ-batch 구성: 풀을 steering 의 Decoder{live_kv, wait} 로 변환 후 steer_decode_set.
// used 를 in-place 공유 → 같은 라운드에서 2회 호출 시 disjoint(advance).
std::vector<int> NodeScheduler::compose_decode_batch(std::vector<char>& used) const {
    std::vector<Decoder> decoders;
    decoders.reserve(pool_.size());
    for (const auto& q : pool_) {
        decoders.push_back(Decoder{q.live_kv(), q.wait});
    }
    return steer_decode_set(decoders, used, op_);
}

// 한 라운드:
//  ① 2 μ-batch(used 공유 disjoint): p1 항상, 풀≥2×decode_count_target 이면 p2.
//  ② used[i] 인 디코더 dec++/wait=0, 아니면 wait++.
//  ③ dec≥dtot retire(완료수 집계), 그 prompt 를 departed hole 로 모음.
//  ④ per-completion 힐링: hole 마다 ideal=hole 로 정확히 1개 admit (toxic-fit, like-for-like).
//     완료한 수만큼만 되채움(hole 단위) — batched 평균 힐링 금지.
//  ⑤ 완료수 반환.
int NodeScheduler::advance_round() {
    // ① 2 μ-batch — used 공유로 disjoint.
    std::vector<char> used(pool_.size(), 0);
    std::vector<int> p1 = compose_decode_batch(used);
    if (static_cast<int>(pool_.size()) >= 2 * op_.decode_count_target) {
        std::vector<int> p2 = compose_decode_batch(used);
        (void)p2;  // 진행은 used 마스크로 일괄 적용 — 선택 인덱스 별도 사용 없음.
    }

    // ② advance: 선택분 dec++/wait=0, 미선택(잉여) wait++.
    for (std::size_t i = 0; i < pool_.size(); ++i) {
        if (used[i]) {
            ++pool_[i].dec;
            pool_[i].wait = 0;
        } else {
            ++pool_[i].wait;
        }
    }

    // ③ retire: dec≥dtot 완료 → hole(prompt) 수집, 나머지 유지.
    std::vector<int> holes;
    std::vector<NodeDecoder> keep;
    keep.reserve(pool_.size());
    for (const auto& q : pool_) {
        if (q.dec >= q.dtot) {
            holes.push_back(q.prompt);
        } else {
            keep.push_back(q);
        }
    }
    pool_.swap(keep);

    // ④ per-completion 힐링: 각 hole 에 ideal=hole 로 정확히 1개 like-for-like 되채움.
    const long long cap_room = node_cap_room(op_);
    for (int hole : holes) {
        PulledRequest r = src_.pull_near(static_cast<double>(hole), cap_room);
        if (r.prompt >= 0) {
            NodeDecoder q;
            q.prompt = r.prompt;
            q.dtot = r.dtot;
            q.dec = 0;
            q.wait = 0;
            pool_.push_back(q);
        }
    }

    // ⑤ 완료수.
    return static_cast<int>(holes.size());
}

long long NodeScheduler::sum_live_kv() const {
    long long s = 0;
    for (const auto& q : pool_) s += q.live_kv();
    return s;
}

double NodeScheduler::mean_live_kv() const {
    if (pool_.empty()) return 0.0;
    return static_cast<double>(sum_live_kv()) / static_cast<double>(pool_.size());
}

}  // namespace puls
