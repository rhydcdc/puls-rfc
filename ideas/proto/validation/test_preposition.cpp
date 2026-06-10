// proto/validation — Preposition 단위검증. CHECKLIST §3 preposition 3 체크.
//  ① 잔여로 완료 슬롯 정확 예측 (slots_left == target − survivors, place 후 done)
//  ② presend 후 두 배치 (62, kv_operating_target) 재센터 (sum ≈ target, 정확 횟수)
//  ③ off-fit 강제분 보정 (강제 LONG 후 ideal 하락, 이후에도 total 수렴).
#include "scheduler/preposition.h"
#include "sim/harness.h"

#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            std::printf("FAIL: %s (line %d)\n", (msg), __LINE__); \
            g_fail = 1;                                           \
        }                                                         \
    } while (0)

using namespace proto;

// ① 잔여로 완료 슬롯 정확 예측.
static void test_slot_prediction(const puls::OperatingPoint& op) {
    const int C = 30;  // survivors
    Preposition pp(0.0, C, op);
    CHECK(pp.slots_left() == op.decode_count_target - C,
          "slots_left == decode_count_target - survivors");
    CHECK(!pp.done(), "not done while slots remain");
    const int to_place = op.decode_count_target - C;
    for (int i = 0; i < to_place; ++i) {
        CHECK(!pp.done(), "not done before last placement");
        pp.place(0);
    }
    CHECK(pp.done(), "done after placing exactly slots_left");
}

// ② presend 후 두 배치 재센터: C survivors summing S, ideal 채움 → sum ≈ kv_operating_target.
static void test_recenter(const puls::OperatingPoint& op) {
    const int C = 40;
    const long long per = (long long)op.ctx_balance;  // survivor ≈ ctx_balance
    const double S = (double)C * (double)per;
    Preposition pp(S, C, op);

    double sum = S;
    int placements = 0;
    while (!pp.done()) {
        double id = pp.next_ideal();
        long long v = std::llround(id);
        pp.place(v);
        sum += (double)v;
        ++placements;
    }
    CHECK(placements == op.decode_count_target - C, "exactly (target - C) placements");
    const double rel = std::fabs(sum - (double)op.kv_operating_target)
                     / (double)op.kv_operating_target;
    CHECK(rel <= 0.01, "final sum within ~1% of kv_operating_target");
}

// ③ off-fit 보정: 첫 슬롯에 강제 LONG → ideal 하락 → 이후에도 total 수렴.
static void test_offfit_correction(const puls::OperatingPoint& op) {
    const int C = 40;
    const long long per = (long long)op.ctx_balance;
    const double S = (double)C * (double)per;
    Preposition pp(S, C, op);

    const double ideal_before = pp.next_ideal();
    const long long forced_long = 5 * (long long)op.ctx_balance;  // off-fit LONG
    pp.place(forced_long);
    double sum = S + (double)forced_long;
    const double ideal_after = pp.next_ideal();
    CHECK(ideal_after < ideal_before, "ideal drops after forced off-fit LONG");

    while (!pp.done()) {
        double id = pp.next_ideal();
        long long v = std::llround(id);
        pp.place(v);
        sum += (double)v;
    }
    const double rel = std::fabs(sum - (double)op.kv_operating_target)
                     / (double)op.kv_operating_target;
    CHECK(rel <= 0.01, "total still converges to kv_operating_target after off-fit");
}

int main() {
    const puls::OperatingPoint op = make_deployment().op;
    test_slot_prediction(op);
    test_recenter(op);
    test_offfit_correction(op);
    if (g_fail) {
        std::printf("test_preposition FAILED\n");
        return 1;
    }
    std::printf("test_preposition OK\n");
    return 0;
}
