// proto/sim — 드라이버 A (BASELINE): 현 node-local steering 설계를 유한 큐 + 멀티턴
//   워크로드 위로 port 한 대조군. CHECKLIST §4 baseline.
//
// 현 설계(puls-engine/sim/lifecycle.cpp 의 노드 로직)를 그대로 재현:
//   잉여 10(decode_pool=134) · 2 μ-batch steer_decode_set(used 공유 disjoint) ·
//   노드 age-cap(op.age_cap) · per-completion ideal=hole 힐링(like-for-like).
// 차이(대조 핸디캡): HBM 캐시 비활성(eligibility=INF → "캐싱 없이 방출") · 글로벌
//   pre-positioning 없음 · 글로벌 age-cap 강제 라우팅 없음. 멀티턴 복귀·SSD 오프로드는
//   드라이버 B 와 동일 공유(공정 대조). 복귀는 SSD 재로드 또는 재계산.
//
// 인구 보존: 도착 = 이탈(대화 종료 시 fresh 1, 복귀는 예약). flood 없음.
//   콜드스타트 = global_scheduler.cpp::cold_start (interleave-greedy, 다양성 보존) —
//   per-node 센터링(all-mid 붕괴)이 아님.
#include "sim/harness.h"
#include "sim/metrics.h"
#include "sim/kpi.h"
#include "scheduler/queue.h"
#include "scheduler/cache.h"
#include "sim/workload_mt.h"
#include "core/steering.h"
#include "core/global_scheduler.h"
#include "core/workload.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace proto;

namespace {

// 한 노드의 상주 디코더. live_kv = prompt + dec (steering 의 kv_length).
struct SimDecoder {
    long long id;
    int prompt;
    int max_tokens;
    int dec;
    int wait;
    int turn;
    int cached_prefix;
    bool met_ttft = false;
    long long live_kv() const { return (long long)prompt + dec; }
};

// prompt 길이 → class: 0 short(<16000) / 1 mid(<256000) / 2 long(else). (분포 B 경계)
inline int cls(int prompt) { return prompt < 16000 ? 0 : (prompt < 256000 ? 1 : 2); }

}  // namespace

int main(int argc, char** argv) {
    // ── 셋업(harness 공유) ───────────────────────────────────────────────────────
    Deployment d = make_deployment();
    auto op = d.op;
    SimConfig cfg;

    const int iters = argc > 1 ? std::atoi(argv[1]) : cfg.iters;
    const int Z     = argc > 2 ? std::atoi(argv[2]) : cfg.Z;
    const int seed_backlog = argc > 6 ? std::atoi(argv[6]) : cfg.seed_backlog;  // 비교 일치용
    const double beta       = argc > 7 ? std::atof(argv[7]) : cfg.contention_beta;       // Plan3 컨텐션 β
    const double offload_bw = argc > 8 ? std::atof(argv[8]) : cfg.offload_bw_bytes_per_round;  // SSD 재로드 BW
    if (argc > 9) op.age_cap = std::atoi(argv[9]);  // 노드 age-cap 스윕(A 의 steer_decode_set 공정성)
    const int warm  = cfg.warm;
    const unsigned seed = cfg.seed;

    int ec = edge_cutoff(op, seed);
    MultiTurnWorkload wl(seed, ec, cfg.mt);
    GlobalQueue queue(cfg.global_age_cap);
    // eligibility_threshold = INT_MAX → HBM 캐싱 비활성(현 설계 = "캐싱 없이 방출").
    ClusterCache cache(Z, node_cache_capacity_bytes(op), d.model.kv_bytes_per_token(),
                       CacheConfig{INT_MAX, cfg.evict_age, cfg.gone_age,
                                   offload_bw});
    PendingReturns pend;
    Metrics M;
    SLO slo{cfg.slo_ttft_us, cfg.slo_tbt_mult * round_us(d)};
    Kpi K;

    const long long cap_room = (long long)op.decode_pool * (long long)op.ctx_balance * 4;

    // admit 마다 TTFT(큐 대기 + 캐시-aware 프리필) → SLO 충족 여부. warm 후만 K 에 누적.
    auto admitTTFT = [&](const SimDecoder& q, long long now, int it) -> bool {
        Request r;
        r.id = q.id; r.prompt = q.prompt; r.cached_prefix = q.cached_prefix;
        r.turn = q.turn; r.arrival = now;
        double wait_us = (double)(now - r.arrival) * round_us(d);
        double comp_us;
        if (r.turn > 0) {
            auto lk = cache.lookup(r.id, now);
            int P2 = r.prompt - r.cached_prefix;
            if (lk.result == CacheResult::Hit)            comp_us = prefill_us(d, P2);
            else if (lk.result == CacheResult::MissReload) comp_us = prefill_us(d, P2) + lk.reload_cost * round_us(d);
            else                                         comp_us = prefill_us(d, r.prompt);
        } else {
            comp_us = prefill_us(d, r.prompt);
        }
        double ttft = wait_us + comp_us;
        return (it >= cfg.warm) ? K.record_ttft(ttft, slo) : (ttft <= slo.t_ttft_us);
    };

    std::vector<std::vector<SimDecoder>> pools(Z);
    long long next_dec_id = 0;  // 콜드스타트 디코더 신규 id 카운터

    // ── 인구 시드: 절대 backlog 만 (A/C 는 콜드스타트를 *자체 draw* 로 채우고 큐는 top-up 만
    //    pull → 노드 채움분을 시드하면 큐가 안 빠져 만성 적체. B 의 post-콜드스타트 큐(~수백)에
    //    맞춰 seed_backlog 만 시드. top-up 여유분(2×Z)을 더해 hole 보충이 굶지 않게. ──
    for (long long i = 0; i < (long long)seed_backlog + 2LL * (long long)Z; ++i)
        queue.push(wl.fresh_arrival(0));

    // ── 콜드스타트: 디코드 풀 = global_scheduler.cpp::cold_start (interleave-greedy,
    //    다양성 보존) 로 분배 → 분포 B(20/70/10) 보존. per-node 센터링(all-mid 붕괴) 아님.
    {
        std::mt19937 init_rng(seed);
        std::vector<int> draw;
        draw.reserve(Z * op.decode_pool * 2);
        for (int i = 0; i < Z * op.decode_pool * 2; ++i)
            draw.push_back(puls::sample_distribution_b(init_rng));
        puls::GateResult g = puls::gate(draw, op);
        std::vector<puls::ClusterNode> cnodes(Z);
        puls::cold_start(g.kept, cnodes, op, seed);

        std::uniform_int_distribution<int> dtot(cfg.mt.dtot_lo, cfg.mt.dtot_hi);
        for (int z = 0; z < Z; ++z) {
            auto& pool = pools[z];
            int take = std::min((int)cnodes[z].lengths.size(), op.decode_pool);
            for (int i = 0; i < take; ++i) {
                int prompt = cnodes[z].lengths[i];
                pool.push_back(SimDecoder{next_dec_id++, prompt, dtot(init_rng),
                                          /*dec*/0, /*wait*/0, /*turn*/0, /*cached_prefix*/0});
                pool.back().met_ttft = admitTTFT(pool.back(), /*now*/0, /*it*/0);
            }
            // 모자라면 ideal=hole 로 보충(기존 디코더 prompt 를 hole 로, like-for-like).
            while ((int)pool.size() < op.decode_pool) {
                int hole = pool.empty()
                               ? (int)op.ctx_balance
                               : pool[init_rng() % pool.size()].prompt;
                Request r = queue.pull_near((double)hole, cap_room);
                if (r.id < 0) break;
                pool.push_back(SimDecoder{r.id, r.prompt, r.max_tokens,
                                          0, 0, r.turn, r.cached_prefix});
                pool.back().met_ttft = admitTTFT(pool.back(), /*now*/0, /*it*/0);
            }
        }
    }

    // ── 정상상태 루프 ────────────────────────────────────────────────────────────
    for (int it = 0; it < iters; ++it) {
        const long long now = it;

        // ① 캐시 evict + due 복귀 push (per-round flood 없음 — 인구 보존).
        cache.evict_tick(now);
        for (auto& r : pend.due(now)) queue.push(r);

        double rResid[3] = {0, 0, 0};
        double rMean = 0;
        int rNodes = 0;

        for (int z = 0; z < Z; ++z) {
            auto& pool = pools[z];

            // ② 디코드 composition: 2 μ-batch steer_decode_set, used 공유 disjoint.
            std::vector<puls::Decoder> decoders;
            decoders.reserve(pool.size());
            for (const auto& q : pool) decoders.push_back(puls::Decoder{q.live_kv(), q.wait});
            std::vector<char> used(pool.size(), 0);

            std::vector<int> p1 = puls::steer_decode_set(decoders, used, op);
            if (it >= warm) {
                long long s = 0;
                int metc = 0;
                for (int i : p1) { s += pool[i].live_kv(); if (pool[i].met_ttft) ++metc; }
                M.record_batch((int)p1.size(), s, op, /*b2*/false);
                K.record_mubatch((int)p1.size(), s, metc, d, slo, beta);
            }
            if ((int)pool.size() >= 2 * op.decode_count_target) {
                std::vector<int> p2 = puls::steer_decode_set(decoders, used, op);
                if (it >= warm) {
                    long long s = 0;
                    int metc = 0;
                    for (int i : p2) { s += pool[i].live_kv(); if (pool[i].met_ttft) ++metc; }
                    M.record_batch((int)p2.size(), s, op, /*b2*/true);
                    K.record_mubatch((int)p2.size(), s, metc, d, slo, beta);
                }
            }

            // ③ advance: 선택분 dec++/wait=0, 미선택 wait++.
            for (std::size_t i = 0; i < pool.size(); ++i) {
                if (used[i]) { ++pool[i].dec; pool[i].wait = 0; }
                else ++pool[i].wait;
            }

            // ④ retire + heal: 완료(dec≥max_tokens) → on_complete + 복귀/신규 + hole 수집.
            std::vector<int> holes;
            {
                std::vector<SimDecoder> keep;
                keep.reserve(pool.size());
                for (const auto& q : pool) {
                    if (q.dec >= q.max_tokens) {
                        cache.on_complete(z, q.id, q.prompt, now);
                        Request asReq;
                        asReq.id            = q.id;
                        asReq.prompt        = q.prompt;
                        asReq.max_tokens    = q.max_tokens;
                        asReq.cached_prefix = q.cached_prefix;
                        asReq.arrival       = now;
                        asReq.turn          = q.turn;
                        auto ret = wl.maybe_return(asReq, now);
                        if (ret.returns) pend.add(ret.return_round, ret.next);
                        else             queue.push(wl.fresh_arrival(now));  // 대화 종료 → 신규 1
                        holes.push_back(q.prompt);  // ideal=hole 되채움용 (like-for-like)
                    } else {
                        keep.push_back(q);
                    }
                }
                pool.swap(keep);
            }
            // hole 마다 정확히 1개 ideal=hole 되채움(toxic-fit). 복귀 턴이면 TTFT 지불.
            // A 는 node-local: 글로벌 age-cap 강제 없음 → pull_near (pull_slot 아님).
            for (int hole : holes) {
                Request r = queue.pull_near((double)hole, cap_room);
                if (r.id < 0) break;  // 소스 고갈 시 차단
                pool.push_back(SimDecoder{r.id, r.prompt, r.max_tokens,
                                          0, 0, r.turn, r.cached_prefix});

                // 복귀(turn>0) → 캐시 조회 비용 = TTFT (큐 대기 포함). A: 캐싱 off → Hit 없음.
                if (r.turn > 0) {
                    auto lk = cache.lookup(r.id, now);
                    int P2 = r.prompt - r.cached_prefix;                          // new message
                    double wait_us = (double)(now - r.arrival) * round_us(d);     // 큐 대기 (A: 글로벌 강제 없음 → 길 수 있음)
                    double comp_us;
                    if (lk.result == CacheResult::Hit)            comp_us = prefill_us(d, P2);
                    else if (lk.result == CacheResult::MissReload) comp_us = prefill_us(d, P2) + lk.reload_cost * round_us(d);
                    else                                         comp_us = prefill_us(d, r.prompt);  // full recompute
                    double ttft = wait_us + comp_us;
                    pool.back().met_ttft = (it >= warm) ? K.record_ttft(ttft, slo) : (ttft <= slo.t_ttft_us);
                    if (it >= warm) {
                        M.ttft.push_back(ttft);
                        M.returns++;
                        if (lk.result == CacheResult::Hit) M.hbm_hit++;
                        else if (lk.result == CacheResult::MissReload) M.ssd_reload++;
                        else M.recompute++;
                        M.saved_rounds += prefill_us(d, r.prompt) - comp_us;     // vs full-recompute baseline
                    }
                } else {
                    // 첫 턴 admit: full prefill TTFT (큐 대기 포함).
                    double ttft = (double)(now - r.arrival) * round_us(d) + prefill_us(d, r.prompt);
                    pool.back().met_ttft = (it >= warm) ? K.record_ttft(ttft, slo) : (ttft <= slo.t_ttft_us);
                }
            }

            // 풀 진단(post-warm 노드 표본): 평균 live_kv + class 상주분율.
            if (it >= warm && !pool.empty()) {
                long long s = 0;
                int rc[3] = {0, 0, 0};
                for (const auto& q : pool) { s += q.live_kv(); ++rc[cls(q.prompt)]; }
                rMean += (double)s / (double)pool.size();
                for (int c = 0; c < 3; ++c) rResid[c] += (double)rc[c] / (double)pool.size();
                ++rNodes;
            }
        }

        if (it >= warm && rNodes > 0) {
            const double frac[3] = {rResid[0] / rNodes, rResid[1] / rNodes, rResid[2] / rNodes};
            M.record_pool(rMean / rNodes, frac);
        }
    }

    // ── 출력 ─────────────────────────────────────────────────────────────────────
    std::printf("[CONFIG-A] ctx_balance=%.0f decode_count_target=%d decode_pool=%d Z=%d "
                "iters=%d eligibility=off evict_age=%d global_age_cap=%d edge_cutoff=%d "
                "beta=%.3f offload_bw=%.3g\n",
                op.ctx_balance, op.decode_count_target, op.decode_pool, Z, iters,
                cfg.evict_age, cfg.global_age_cap, ec, beta, offload_bw);
    M.print_summary("BASELINE-A", op);
    K.sim_us = (double)(iters - cfg.warm) * round_us(d);
    K.print("A", slo);
    std::printf("[QUEUE-A] size=%d max_wait=%lld forced=%lld (A: 글로벌 age-cap 없음 → starvation 지표)\n",
                queue.size(), queue.max_wait(iters), queue.forced_count());
    return 0;
}
