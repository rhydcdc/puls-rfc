// proto/scheduler — 글로벌 풀 = 유한 큐 + 글로벌 age-cap (pull 시점 강제).
// CHECKLIST §3 queue. 순수 스케줄러 로직: 분포B·RNG 없음(요청은 sim 이 push).
#pragma once
#include <vector>
#include "scheduler/request.h"

namespace proto {

// 도착 타임스탬프 달린 유한 큐.
//  - pull_near : 평범한 길이-fit 최근접 (A 힐링 / B 일반 슬롯).
//  - pull_slot : 글로벌 age-cap 인지 — wait(now−arrival) > cap 이면 가장 오래된 것을 *강제*
//                (forced_++), 아니면 pull_near 와 동일. (steering age-cap 의 글로벌 판.)
// 강제는 슬롯 1개당 pull 1회로 자연 상한된다(폐기 없음).
class GlobalQueue {
public:
    explicit GlobalQueue(int global_age_cap);

    void push(const Request& r);

    // ideal_prompt 에 |prompt − ideal| 최소인 (prompt ≤ cap_room) 요청을 빼서 반환. 없으면 id<0.
    Request pull_near(double ideal_prompt, long long cap_room);

    // 글로벌 age-cap 슬롯 채움: wait > age_cap 인 게 있으면 가장 오래된 것 강제(cap_room 무시,
    // off-fit 주입), 없으면 pull_near(ideal, cap_room). 비었으면 id<0.
    Request pull_slot(double ideal_prompt, long long cap_room, long long now);

    int       size() const { return (int)q_.size(); }
    long long forced_count() const { return forced_; }
    long long max_wait(long long now) const;   // 최장 대기 (starvation 진단)

private:
    std::vector<Request> q_;
    int       age_cap_;
    long long forced_ = 0;

    int find_near(double ideal_prompt, long long cap_room) const;  // 인덱스(-1 없음)
    int find_oldest_aged(long long now) const;                     // wait>cap 중 최古 (-1 없음)
    Request take(int idx);                                         // idx 제거 후 반환
};

} // namespace proto
