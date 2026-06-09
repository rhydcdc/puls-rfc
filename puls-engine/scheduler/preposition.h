// proto/scheduler — 결정론 pre-positioning composer. CHECKLIST §3 preposition.
// 한 배치(62 슬롯)를 (decode_count_target, kv_operating_target) 로 재센터.
// 순수(큐 비의존): 빠진 슬롯마다 ideal 을 순차 산출, 실제 채운 길이를 place() 로 반영.
#pragma once
#include "core/operating_point.h"

namespace proto {

// 결정론 재센터 상태기. 노드 steering 대체(글로벌 1회 결정).
//   ideal = (kv_operating_target − sum) / slots_left.
// 드라이버 사용:
//   Preposition pp(survivors_sum_kv, survivors_count, op);
//   while (!pp.done()) { double id = pp.next_ideal(); /* queue.pull_near(id) */
//                        pp.place(pulled.live_kv); /* 슬롯 채움 */ }
class Preposition {
public:
    Preposition(double survivors_sum_kv, int survivors_count, const puls::OperatingPoint& op);

    bool   done() const { return slots_left_ <= 0; }
    double next_ideal() const;              // 다음 슬롯 ideal 길이 (slots_left>0 가정)
    void   place(long long placed_live_kv); // 실제 채운 길이 반영 (sum +=, slots_left --)
    int    slots_left() const { return slots_left_; }

private:
    double    sum_;
    int       slots_left_;
    long long kv_target_;
};

} // namespace proto
