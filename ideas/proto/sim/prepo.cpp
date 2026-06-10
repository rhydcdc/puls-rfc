// proto/sim — 드라이버 B: 글로벌 결정론 pre-positioning.
// CHECKLIST §4 prepo. 노드 = 실행기(124 = 2×decode_count_target 상주), 노드 steering·
// 잉여·노드 힐링·노드 age-cap 전부 없음. 정상상태 = scheduler/preposition 결정론 presend
// + 글로벌 age-cap 강제(pull_slot). on-node 캐시 + 멀티턴 복귀 히트 라우팅(holder-node 암묵).
// 인구 보존: 도착 = 이탈 (완료마다 복귀 예약 또는 fresh 1). flood 없음.
// 근거: ../predictive_prepositioning_scheduler.md, ../CHECKLIST.md §4 prepo.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "scheduler/cache.h"
#include "scheduler/preposition.h"
#include "scheduler/queue.h"
#include "sim/harness.h"
#include "sim/kpi.h"
#include "sim/metrics.h"
#include "sim/workload_mt.h"

using namespace proto;

// 노드 상주 디코더. live_kv = prompt + dec (현 KV footprint 근사).
struct Decoder {
    long long id;
    int       prompt;
    int       max_tokens;
    int       dec;
    int       turn;
    int       cached_prefix;
    long long arrival;
    bool      met_ttft = false;
    long long live_kv() const { return (long long)prompt + (long long)dec; }
};

int main(int argc, char** argv) {
    Deployment d = make_deployment();
    const puls::OperatingPoint& op = d.op;
    SimConfig cfg;

    // CLI: [iters] [Z] [eligibility_threshold] [evict_age] [global_age_cap] (기본 = SimConfig).
    const int iters         = (argc > 1) ? std::atoi(argv[1]) : cfg.iters;
    const int Z             = (argc > 2) ? std::atoi(argv[2]) : cfg.Z;
    const int eligibility   = (argc > 3) ? std::atoi(argv[3]) : cfg.eligibility_threshold;
    const int evict_age     = (argc > 4) ? std::atoi(argv[4]) : cfg.evict_age;
    const int global_age_cap = (argc > 5) ? std::atoi(argv[5]) : cfg.global_age_cap;
    const int seed_backlog  = (argc > 6) ? std::atoi(argv[6]) : cfg.seed_backlog;
    const double beta       = (argc > 7) ? std::atof(argv[7]) : cfg.contention_beta;
    const double offload_bw = (argc > 8) ? std::atof(argv[8]) : cfg.offload_bw_bytes_per_round;
    const int gone_age      = cfg.gone_age;
    const unsigned seed     = cfg.seed;

    const int ec = edge_cutoff(op, seed);
    MultiTurnWorkload wl(seed, ec, cfg.mt);
    GlobalQueue queue(global_age_cap);
    // B 는 잉여 없음(124 상주) → 잉여 KV 만큼 캐시 용량이 더 크다(134 기반 A·C 대비 충실한 비대칭).
    ClusterCache cache(Z, node_cache_capacity_bytes(op) + surplus_kv_bytes(d),
                       d.model.kv_bytes_per_token(),
                       {eligibility, evict_age, gone_age, offload_bw});
    PendingReturns pend;
    Metrics M;
    SLO slo{cfg.slo_ttft_us, cfg.slo_tbt_mult * round_us(d)};
    Kpi K;

    const long long cap_room = (long long)op.decode_pool * (long long)op.ctx_balance * 4;

    // 노드 = 2 배치 × decode_count_target. (NO surplus, NO node steering/healing/age-cap.)
    const int NB = op.decode_count_target;  // batch 정원 = 62
    std::vector<std::array<std::vector<Decoder>, 2>> nodes(Z);
    for (auto& n : nodes) {
        n[0].reserve(NB);
        n[1].reserve(NB);
    }

    // admit 마다 TTFT(큐 대기 포함, B: 글로벌 age-cap 이 상한) → SLO 충족 여부 반환.
    //   첫 턴 = full prefill / 복귀 턴 = 캐시-aware(hit/reload/recompute). K 는 전 요청, M 은 복귀만.
    auto admitTTFT = [&](const Request& req, long long now, int it) -> bool {
        double wait_us = (double)(now - req.arrival) * round_us(d);   // 큐 대기 (B: 글로벌 age-cap 이 상한)
        double comp_us;
        if (req.turn > 0) {
            auto lk = cache.lookup(req.id, now);
            int P2 = req.prompt - req.cached_prefix;
            if (lk.result == CacheResult::Hit)             comp_us = prefill_us(d, P2);
            else if (lk.result == CacheResult::MissReload) comp_us = prefill_us(d, P2) + lk.reload_cost * round_us(d);
            else                                           comp_us = prefill_us(d, req.prompt);
            double ttft = wait_us + comp_us;
            bool met = (it >= cfg.warm) ? K.record_ttft(ttft, slo) : (ttft <= slo.t_ttft_us);
            if (it >= cfg.warm) {
                M.ttft.push_back(ttft);
                M.returns++;
                if (lk.result == CacheResult::Hit)             M.hbm_hit++;
                else if (lk.result == CacheResult::MissReload) M.ssd_reload++;
                else                                           M.recompute++;
                M.saved_rounds += prefill_us(d, req.prompt) - comp_us;
            }
            return met;
        }
        // 첫 턴 admit: full prefill.
        double ttft = wait_us + prefill_us(d, req.prompt);
        return (it >= cfg.warm) ? K.record_ttft(ttft, slo) : (ttft <= slo.t_ttft_us);
    };

    // ── Cold start: 큐를 인구 보존 시드로 채우고(아래) 각 노드 2 배치를 결정론 124 센터 ──
    //   now=0 이라 aged 없음 → pull_near (강제 무의미).
    // 인구 = 노드 채움(Z×node_min) + 절대 backlog. backlog 이 노드용량 배수면 영구 적체→강제 폭주.
    const long long seed_pop = (long long)Z * (long long)op.node_min + (long long)seed_backlog;
    for (long long i = 0; i < seed_pop; ++i) queue.push(wl.fresh_arrival(0));
    for (int z = 0; z < Z; ++z)
        for (int b = 0; b < 2; ++b) {
            std::vector<Decoder>& batch = nodes[z][b];
            Preposition pp(0.0, 0, op);
            while (!pp.done()) {
                double id = pp.next_ideal();
                Request r = queue.pull_near(id, cap_room);
                if (r.id < 0) break;
                batch.push_back(Decoder{r.id, r.prompt, r.max_tokens, 0, r.turn, r.cached_prefix, r.arrival});
                batch.back().met_ttft = admitTTFT(r, /*now*/0, /*it*/0);
                pp.place(r.prompt);
            }
        }

    // ── 라운드 루프 ──────────────────────────────────────────────────────────────
    for (int it = 0; it < iters; ++it) {
        const long long now = it;

        // 1. 캐시 정리 · 만기 복귀 큐 투입. (flood 없음 — fresh 도착은 완료 시 1:1.)
        cache.evict_tick(now);
        for (auto& r : pend.due(now)) queue.push(r);

        // 2. 노드별·배치별: 디코드 스텝 기록 → retire → 결정론 pre-position refill.
        for (int z = 0; z < Z; ++z)
            for (int b = 0; b < 2; ++b) {
                std::vector<Decoder>& batch = nodes[z][b];

                // a. decode 스텝 + 기록 (디코드 전 현재 composition).
                long long sum_kv = 0;
                int metc = 0;
                for (const auto& dc : batch) { sum_kv += dc.live_kv(); if (dc.met_ttft) ++metc; }
                if (it >= cfg.warm) {
                    M.record_batch((int)batch.size(), sum_kv, op, /*b2=*/(b == 1));
                    K.record_mubatch((int)batch.size(), sum_kv, metc, d, slo, beta);
                }
                for (auto& dc : batch) ++dc.dec;

                // b. retire: dec >= max_tokens → 완료(캐시 기록 + 복귀 예약 또는 fresh 도착).
                std::vector<Decoder> survivors;
                survivors.reserve(batch.size());
                for (const auto& dc : batch) {
                    if (dc.dec >= dc.max_tokens) {
                        const int total_len = dc.prompt + dc.dec;  // 누적 KV 길이
                        cache.on_complete(z, dc.id, total_len, now);
                        Request asReq;
                        asReq.id            = dc.id;
                        asReq.prompt        = dc.prompt;
                        asReq.max_tokens    = dc.max_tokens;
                        asReq.cached_prefix = dc.cached_prefix;
                        asReq.arrival       = now;
                        asReq.turn          = dc.turn;
                        auto ret = wl.maybe_return(asReq, now);
                        if (ret.returns) pend.add(ret.return_round, ret.next);  // 복귀 예약
                        else             queue.push(wl.fresh_arrival(now));     // 신규 도착(인구 보존)
                    } else {
                        survivors.push_back(dc);
                    }
                }
                batch.swap(survivors);

                // c. 결정론 pre-position refill: 빈 슬롯마다 pull_slot(글로벌 age-cap 강제).
                double S = 0.0;
                for (const auto& dc : batch) S += (double)dc.live_kv();
                Preposition pp(S, (int)batch.size(), op);
                while (!pp.done()) {
                    double id = pp.next_ideal();
                    Request r = queue.pull_slot(id, cap_room, now);
                    if (r.id < 0) break;  // 큐 고갈
                    batch.push_back(Decoder{r.id, r.prompt, r.max_tokens, 0, r.turn, r.cached_prefix, r.arrival});
                    batch.back().met_ttft = admitTTFT(r, now, it);  // 전 요청 TTFT(복귀 턴은 캐시 비용 지불)
                    pp.place(r.prompt);   // 강제 off-fit 길이는 다음 슬롯 ideal 이 자가보정(composer)
                }
            }

        // 3. warm 후 풀 진단: 124 평균 live_kv + class 상주분율(short/mid/long).
        if (it >= cfg.warm) {
            long long total = 0;
            int n = 0;
            double cnt[3] = {0, 0, 0};
            for (int z = 0; z < Z; ++z)
                for (int b = 0; b < 2; ++b)
                    for (const auto& dc : nodes[z][b]) {
                        total += dc.live_kv();
                        ++n;
                        if (dc.prompt < 16000)       cnt[0] += 1.0;
                        else if (dc.prompt < 256000) cnt[1] += 1.0;
                        else                          cnt[2] += 1.0;
                    }
            if (n > 0) {
                double frac[3] = {cnt[0] / n, cnt[1] / n, cnt[2] / n};
                M.record_pool((double)total / (double)n, frac);
            }
        }
    }

    M.forced = queue.forced_count();

    std::printf(
        "[CONFIG] ctx_balance=%.0f decode_count_target=%d pool=%d Z=%d iters=%d "
        "eligibility=%d evict_age=%d gone_age=%d global_age_cap=%d edge_cutoff=%d "
        "beta=%.2f offload_bw=%.3g\n",
        op.ctx_balance, op.decode_count_target, 2 * NB, Z, iters,
        eligibility, evict_age, gone_age, global_age_cap, ec,
        beta, offload_bw);
    M.print_summary("PREPO-B", op);
    K.sim_us = (double)(iters - cfg.warm) * round_us(d);
    K.print("B", slo);
    std::printf("[QUEUE] size=%d max_wait=%lld forced=%lld\n",
                queue.size(), queue.max_wait(iters), queue.forced_count());
    return 0;
}
