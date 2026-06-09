// proto/scheduler — 글로벌 풀 구현. CHECKLIST §3 queue.
// pull_near = 길이-fit 최근접. pull_slot = 글로벌 age-cap(가장 오래된 것 강제) → pull_near.
// 강제는 슬롯당 1회라 자연 상한(폐기 없음). 무축출 — 큐는 push/pull 만.
#include "scheduler/queue.h"

#include <cmath>

namespace proto {

GlobalQueue::GlobalQueue(int global_age_cap) : age_cap_(global_age_cap) {}

void GlobalQueue::push(const Request& r) { q_.push_back(r); }

// |prompt − ideal| 최소인 (prompt ≤ cap_room) 인덱스. 없으면 -1.
int GlobalQueue::find_near(double ideal_prompt, long long cap_room) const {
    int best = -1;
    double best_dist = 1e18;
    for (int i = 0; i < (int)q_.size(); ++i) {
        if ((long long)q_[i].prompt > cap_room) continue;
        const double dist = std::fabs((double)q_[i].prompt - ideal_prompt);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

// wait(now − arrival) > age_cap 중 가장 오래된(최소 arrival) 인덱스. 없으면 -1.
int GlobalQueue::find_oldest_aged(long long now) const {
    int best = -1;
    long long oldest = 0;
    for (int i = 0; i < (int)q_.size(); ++i) {
        if (now - q_[i].arrival > age_cap_) {
            if (best < 0 || q_[i].arrival < oldest) {
                oldest = q_[i].arrival;
                best = i;
            }
        }
    }
    return best;
}

// idx 제거(swap-with-last) 후 반환.
Request GlobalQueue::take(int idx) {
    Request r = q_[idx];
    q_[idx] = q_.back();
    q_.pop_back();
    return r;
}

Request GlobalQueue::pull_near(double ideal_prompt, long long cap_room) {
    int idx = find_near(ideal_prompt, cap_room);
    if (idx < 0) return Request{};
    return take(idx);
}

Request GlobalQueue::pull_slot(double ideal_prompt, long long cap_room, long long now) {
    int aged = find_oldest_aged(now);
    if (aged >= 0) {
        ++forced_;
        return take(aged);   // 강제: cap_room 무시(off-fit 주입)
    }
    return pull_near(ideal_prompt, cap_room);
}

long long GlobalQueue::max_wait(long long now) const {
    long long mx = 0;
    for (const auto& r : q_) {
        const long long w = now - r.arrival;
        if (w > mx) mx = w;
    }
    return mx;
}

} // namespace proto
