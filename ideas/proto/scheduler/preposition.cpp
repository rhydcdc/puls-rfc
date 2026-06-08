// proto/scheduler — 결정론 pre-positioning composer 구현. CHECKLIST §3 preposition.
// 근거: predictive_prepositioning_scheduler.md "글로벌 1회 결정 재센터".
// 한 배치(decode_count_target 슬롯)를 (decode_count_target, kv_operating_target) 로 재센터.
// 순수(큐 비의존): 빠진 슬롯마다 ideal = (target − sum)/slots_left 산출, 실제 채운 길이를 place().
#include "scheduler/preposition.h"

namespace proto {

Preposition::Preposition(double survivors_sum_kv, int survivors_count,
                         const puls::OperatingPoint& op)
    : sum_(survivors_sum_kv),
      slots_left_(op.decode_count_target - survivors_count),
      kv_target_(op.kv_operating_target) {}

// 다음 슬롯 ideal 길이 = 남은 KV 예산 / 남은 슬롯 수. (slots_left_ > 0 가정.)
double Preposition::next_ideal() const {
    return (double)(kv_target_ - sum_) / (double)slots_left_;
}

// 실제 채운 길이 반영: sum 누적, 슬롯 하나 소비.
void Preposition::place(long long placed_live_kv) {
    sum_ += (double)placed_live_kv;
    --slots_left_;
}

}  // namespace proto
