// proto/validation — workload_mt 단위검증. CHECKLIST §4 workload_mt 4 체크:
//   ① fresh arrival 유효(게이팅·turn0·prefix0·max_tokens 범위·id 단조증가)
//   ② 복귀확률 길이의존(short < long)
//   ③ 복귀 시 길이 누적(prompt↑·cached_prefix·turn+1·id 유지·arrival)
//   ④ 평균 턴수 ≈ 2.5 (느슨한 다중턴 밴드, 실측값 보고)
#include "sim/workload_mt.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL: %s (line %d)\n", (msg), __LINE__);         \
            g_fail = 1;                                                   \
        }                                                                 \
    } while (0)

using namespace proto;

// short<mid<long 복귀확률 명확 정렬 + 작은 구체 범위.
static MultiTurnConfig make_cfg() {
    MultiTurnConfig c;
    c.p_return_short = 0.2;
    c.p_return_mid   = 0.6;
    c.p_return_long  = 0.8;
    c.dtot_lo        = 100;
    c.dtot_hi        = 200;
    c.new_msg_lo     = 50;
    c.new_msg_hi     = 150;
    c.think_gap_lo   = 3;
    c.think_gap_hi   = 7;
    return c;
}

// 주어진 prompt class 의 prev 를 만든다(maybe_return 의 복귀확률 분기 검증용).
static Request prev_of_prompt(int prompt) {
    Request r;
    r.id         = 1;
    r.prompt     = prompt;
    r.max_tokens = 150;
    r.turn       = 0;
    r.arrival    = 0;
    return r;
}

int main() {
    const int edge_cutoff = 256000; // long 일부 게이팅 통과(분포 B 상한 1M).
    MultiTurnConfig cfg = make_cfg();

    // ── ① fresh arrival 유효 ──────────────────────────────────────────────
    {
        MultiTurnWorkload w(12345u, edge_cutoff, cfg);
        long long prev_id = -1;
        for (int i = 0; i < 5000; i++) {
            long long now = 1000 + i;
            Request r = w.fresh_arrival(now);
            CHECK(r.prompt <= edge_cutoff, "fresh prompt within edge cutoff");
            CHECK(r.prompt >= 1, "fresh prompt positive");
            CHECK(r.turn == 0, "fresh turn == 0");
            CHECK(r.cached_prefix == 0, "fresh cached_prefix == 0");
            CHECK(r.max_tokens >= cfg.dtot_lo && r.max_tokens <= cfg.dtot_hi,
                  "fresh max_tokens within [dtot_lo,dtot_hi]");
            CHECK(r.arrival == now, "fresh arrival == now");
            CHECK(r.id == prev_id + 1, "fresh id unique increasing");
            prev_id = r.id;
        }
    }

    // ── ② 복귀확률 길이의존: short-class < long-class ─────────────────────
    {
        MultiTurnWorkload w(777u, edge_cutoff, cfg);
        const int N = 20000;
        Request short_prev = prev_of_prompt(5000);   // class 0
        Request long_prev  = prev_of_prompt(300000); // class 2
        int short_ret = 0, long_ret = 0;
        for (int i = 0; i < N; i++)
            if (w.maybe_return(short_prev, 0).returns) short_ret++;
        for (int i = 0; i < N; i++)
            if (w.maybe_return(long_prev, 0).returns) long_ret++;
        double short_rate = (double)short_ret / N;
        double long_rate  = (double)long_ret / N;
        std::printf("return rate: short=%.3f long=%.3f\n", short_rate, long_rate);
        CHECK(short_rate < long_rate, "return rate short < long");
        // 통계 허용오차(±0.03) 내 설정확률 근사.
        CHECK(short_rate > cfg.p_return_short - 0.03 &&
              short_rate < cfg.p_return_short + 0.03,
              "short rate near p_return_short");
        CHECK(long_rate > cfg.p_return_long - 0.03 &&
              long_rate < cfg.p_return_long + 0.03,
              "long rate near p_return_long");
    }

    // ── ③ 복귀 시 길이 누적 ───────────────────────────────────────────────
    {
        MultiTurnWorkload w(999u, edge_cutoff, cfg);
        Request prev = prev_of_prompt(300000); // long class → 높은 복귀확률
        prev.turn = 4;
        prev.id   = 42;
        long long now = 5000;
        int checked = 0;
        for (int i = 0; i < 1000 && checked < 50; i++) {
            auto ret = w.maybe_return(prev, now);
            if (!ret.returns) continue;
            checked++;
            const Request& nx = ret.next;
            CHECK(nx.prompt > prev.prompt, "return prompt accumulates");
            CHECK(nx.cached_prefix == prev.prompt + prev.max_tokens,
                  "return cached_prefix == prev.prompt + prev.max_tokens");
            CHECK(nx.turn == prev.turn + 1, "return turn == prev.turn + 1");
            CHECK(nx.id == prev.id, "return id == prev.id");
            CHECK(ret.return_round == now + (ret.return_round - now),
                  "return_round consistent");
            long long gap = ret.return_round - now;
            CHECK(gap >= cfg.think_gap_lo && gap <= cfg.think_gap_hi,
                  "think_gap within range");
            CHECK(nx.arrival == ret.return_round, "next.arrival == return_round");
            // prompt = cached_prefix + P2, P2 in [new_msg_lo,new_msg_hi].
            int p2 = nx.prompt - nx.cached_prefix;
            CHECK(p2 >= cfg.new_msg_lo && p2 <= cfg.new_msg_hi,
                  "P2 within [new_msg_lo,new_msg_hi]");
        }
        CHECK(checked > 0, "at least one return observed");
    }

    // ── ④ 평균 턴수 ≈ 2.5 ────────────────────────────────────────────────
    {
        MultiTurnWorkload w(2024u, edge_cutoff, cfg);
        const int CHAINS = 20000;
        long long total_turns = 0;
        long long now = 0;
        for (int c = 0; c < CHAINS; c++) {
            Request cur = w.fresh_arrival(now);
            int turns = 1; // 첫 턴
            for (;;) {
                auto ret = w.maybe_return(cur, now);
                if (!ret.returns) break;
                turns++;
                cur = ret.next;
                now = ret.return_round;
            }
            total_turns += turns;
            now += 1;
        }
        double avg = (double)total_turns / CHAINS;
        std::printf("average turns = %.4f (over %d chains)\n", avg, CHAINS);
        CHECK(avg >= 2.0 && avg <= 3.2,
              "average turns in plausible multi-turn band [2.0,3.2]");
    }

    if (g_fail) {
        std::printf("test_workload_mt FAILED\n");
        return 1;
    }
    std::printf("test_workload_mt OK\n");
    return 0;
}
