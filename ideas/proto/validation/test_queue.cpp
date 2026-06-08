// proto/validation — test_queue. CHECKLIST §3 queue 4 체크:
//  ① 글로벌 age-cap: pull_slot 이 wait>cap 인 가장 오래된 것 강제 + forced 카운트 (비-aged 잔류)
//  ② 길이-fit 최근접 + cap_room (cap 초과 건너뜀, 나머지 중 최근접)
//  ③ FIFO 공정성 (어정쩡한 길이도 cap 넘으면 pull_slot 이 반드시 빼냄 — starvation 0)
//  ④ 대용량 시 길이 고갈 0 (특정 길이대 반복 pull 에도 근접 매치 지속)
#include "scheduler/queue.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using proto::GlobalQueue;
using proto::Request;

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);\
            return 1;                                                     \
        }                                                                 \
    } while (0)

// id·prompt·arrival 만 세팅하는 헬퍼.
static Request mk(long long id, int prompt, long long arrival) {
    Request r;
    r.id = id;
    r.prompt = prompt;
    r.arrival = arrival;
    return r;
}

// ── ① 글로벌 age-cap: pull_slot 강제 + 카운트 ───────────────────────────────
static int test_age_cap_forced() {
    GlobalQueue q(/*global_age_cap=*/10);
    // now = 100. wait = now − arrival. cap = 10 → wait > 10 (arrival < 90) 이면 aged.
    q.push(mk(1, 50000, 80));   // wait 20 > 10 → aged
    q.push(mk(2, 60000, 50));   // wait 50 > 10 → aged (더 오래됨)
    q.push(mk(3, 70000, 95));   // wait 5  ≤ 10 → 잔류
    q.push(mk(4, 80000, 90));   // wait 10 = cap → 잔류 (strict >)
    CHECK(q.size() == 4, "initial size");
    CHECK(q.max_wait(100) == 50, "max_wait = oldest wait (id 2, 50)");

    // pull_slot: ideal 이 무엇이든 aged 가 있으면 *가장 오래된* 것(id 2, arrival 50) 강제.
    Request f1 = q.pull_slot(/*ideal*/70000.0, /*cap*/1000000, /*now*/100);
    CHECK(f1.id == 2, "oldest aged forced first (id 2)");
    CHECK(q.forced_count() == 1, "forced_count == 1");

    Request f2 = q.pull_slot(70000.0, 1000000, 100);
    CHECK(f2.id == 1, "next oldest aged forced (id 1)");
    CHECK(q.forced_count() == 2, "forced_count == 2");

    // 더는 aged 없음 → pull_slot 이 길이-fit 최근접으로 fallback (id 3: 70000 에 정확).
    Request near = q.pull_slot(70000.0, 1000000, 100);
    CHECK(near.id == 3, "no aged left → falls back to nearest-fit (id 3)");
    CHECK(q.forced_count() == 2, "forced unchanged on non-forced pull");
    CHECK(q.size() == 1, "one remains (id 4)");
    return 0;
}

// ── ② 길이-fit 최근접 + cap_room ────────────────────────────────────────────
static int test_length_fit_nearest() {
    GlobalQueue q(/*global_age_cap=*/1000);
    q.push(mk(1, 1000, 0));
    q.push(mk(2, 5000, 0));
    q.push(mk(3, 9000, 0));    // cap_room 8000 초과 → 건너뜀
    q.push(mk(4, 6000, 0));

    // ideal=5500, cap_room=8000 → 후보 {1000,5000,6000}, |·−5500| 최소 = 5000(dist500) vs 6000(dist500).
    // 동률 시 먼저 스캔된 것(5000, id=2) 선택.
    Request r = q.pull_near(5500.0, 8000);
    CHECK(r.id == 2, "nearest to 5500 under cap is id 2 (prompt 5000)");
    CHECK(q.size() == 3, "one removed");

    // 이제 ideal=6200, cap_room=8000 → 후보 {1000,9000>cap,6000} → 6000(id4) 최근접.
    Request r2 = q.pull_near(6200.0, 8000);
    CHECK(r2.id == 4, "nearest to 6200 under cap is id 4 (prompt 6000)");

    // cap_room 매우 작아 모두 초과 → 없음(id < 0).
    Request none = q.pull_near(500.0, 100);
    CHECK(none.id < 0, "no candidate under tiny cap returns id<0");
    CHECK(q.size() == 2, "failed pull does not remove");

    // cap_room 충분히 크게 두면 9000(id3) 도 잡힌다(skip 이 cap 때문임을 확인).
    Request big = q.pull_near(9000.0, 100000);
    CHECK(big.id == 3, "over-cap item becomes pickable once cap raised");
    return 0;
}

// ── ③ FIFO 공정성 — 어정쩡한 길이도 cap 넘으면 반드시 빠짐 ─────────────────
static int test_fifo_fairness() {
    GlobalQueue q(/*global_age_cap=*/5);
    // 일찍 도착했지만 길이가 어정쩡해 pull_near 가 절대 안 고르는 요청.
    q.push(mk(99, 7777, 0));      // 어정쩡한 길이, arrival 0
    // 항상 더 좋은 length-fit 인 요청들을 계속 넣고 빼도 99 는 안 뽑힌다 (cap 내라 강제 X).
    for (int t = 1; t <= 5; ++t) {
        q.push(mk(t, 100000, t));                 // ideal=100000 에 완벽 fit
        Request r = q.pull_slot(100000.0, 1000000, /*now*/t);  // wait(99)=t≤5 → 강제 아님
        CHECK(r.id == t, "perfect-fit preferred while id 99 within age-cap");
    }
    CHECK(q.size() == 1, "only the never-fit request remains");
    // wait 가 cap 을 넘는 순간 pull_slot 이 강제로 빼낸다(starvation 0).
    Request forced = q.pull_slot(100000.0, 1000000, /*now=*/100);  // wait = 100 − 0 = 100 > 5
    CHECK(forced.id == 99, "starved request forced out via age-cap");
    CHECK(q.size() == 0, "queue empty after forced pull");
    CHECK(q.forced_count() == 1, "forced counted");
    return 0;
}

// ── ④ 대용량 시 길이 고갈 0 ─────────────────────────────────────────────────
static int test_no_depletion_at_scale() {
    GlobalQueue q(/*global_age_cap=*/1000000);
    // short / mid / long 3 밴드 대용량 투입. 결정론(RNG 없음): id 로 밴드 순환.
    const int per_band = 3000;
    long long id = 0;
    for (int i = 0; i < per_band; ++i) {
        q.push(mk(id++, 2000 + (i % 500), 0));    // short ~2000
        q.push(mk(id++, 50000 + (i % 500), 0));   // mid   ~50000
        q.push(mk(id++, 100000 + (i % 500), 0));  // long  ~100000
    }
    CHECK(q.size() == per_band * 3, "scale population pushed");

    // mid 밴드를 반복 pull — 매번 ideal 에 근접(고갈 없음).
    const double ideal = 50250.0;  // mid 밴드 중심부
    const int pulls = 500;
    for (int k = 0; k < pulls; ++k) {
        Request r = q.pull_near(ideal, 1000000);
        CHECK(r.id >= 0, "pull never runs dry");
        CHECK(std::fabs(static_cast<double>(r.prompt) - ideal) <= 300.0,
              "pulled prompt stays near requested band");
    }
    CHECK(q.size() == per_band * 3 - pulls, "exactly pulls removed");
    return 0;
}

int main() {
    if (test_age_cap_forced()) { std::fprintf(stderr, "test_age_cap_forced failed\n"); return 1; }
    if (test_length_fit_nearest()) { std::fprintf(stderr, "test_length_fit_nearest failed\n"); return 1; }
    if (test_fifo_fairness()) { std::fprintf(stderr, "test_fifo_fairness failed\n"); return 1; }
    if (test_no_depletion_at_scale()) { std::fprintf(stderr, "test_no_depletion_at_scale failed\n"); return 1; }
    std::printf("test_queue OK\n");
    return 0;
}
