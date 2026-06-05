// 검증 — ② 노드 스케줄러(node_scheduler + steering). CONTRACT.md §3-② / §4 / §7.
// 앵커: derive_operating_point(Llama70B, B200, 128) 동작점 위에서
//   1) 센터링 admit 수렴   — admit_centered(true) 로 풀을 채우면 mean_live_kv → ctx_balance (±10%).
//   2) 풀 유지            — advance_round 여러 라운드 후 pool().size() 가 decode_pool 근처.
//   3) per-completion 힐링 toxic-fit 보존 — 상주 긴 요청(prompt≥256K) 비율 초기 vs 후기 보존
//                          (batched 였다면 0 으로 굶음 → 보존되면 per-completion 정상).
//   4) drift 없음         — warmup 후 mean_live_kv 가 라운드 지나도 ctx_balance 근처 안정(early≈late).
//
// 주의(구현 동작에 맞춘 검증):
//   NodeScheduler 는 RNG 없음 — 분포는 src(RequestSource)가 제공.
//   WorkloadSource 는 dtot ≈ prompt(≥1000) 로 두므로, advance_round 가 dec 를 라운드당 +1 만
//   올리는 한 완료(dec≥dtot)는 수만 라운드 뒤에야 난다 = 실질적으로 완료 churn 이 안 보인다.
//   따라서 per-completion 힐링(check 3)·완료 기반 풀 되채움은 실제 분포 B 로는 관측 불가.
//   → check 3 은 **작은 dtot 합성 source**(SmallDtotSource)로 완료를 유도해 검증한다.
//     이 source 는 best-of-K 로 ideal 근방을 고르되 dtot 을 작게 둔다 — node_scheduler.cpp
//     수정 없이(요청 출처만 교체) 완료를 만들어 per-completion 힐링의 like-for-like 를 관측.
//   check 1/2/4 는 실 WorkloadSource(완료 거의 없음)로 센터링/steering 안정성을 본다.
#include "test_framework.h"
#include "core/derive.h"
#include "core/node_scheduler.h"
#include "core/workload.h"
#include "core/request_source.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace puls;

// ── 합성 source: ideal=hole 힐링의 like-for-like 를 그대로 통과시키는 best-of-K source. ──
// node_scheduler.cpp 는 수정 금지 — 완료 churn 을 보이게 하려고 요청 출처만 바꾼다.
//   prompt = best-of-K 로 ideal_tokens 최근접(=WorkloadSource 와 동형, 무한풀 emulation).
//   dtot   = dec_horizon 고정(작음) → 각 요청은 dec_horizon 라운드 안에 완료(dec≥dtot).
// 핵심: 힐링이 ideal=hole(긴 hole)로 pull 하면 이 source 는 긴 교체를 반환한다 = toxic-fit.
//      (batched 라면 ideal≈평균(~100K)이라 긴 거 안 뽑혀 굶음 — 아래 batched 시뮬로 대조.)
class SmallDtotSource : public RequestSource {
public:
    SmallDtotSource(unsigned seed, int dec_horizon, int best_of_k = 200)
        : rng_(seed), horizon_(dec_horizon), k_(best_of_k) {}

    PulledRequest pull_near(double ideal_tokens, long long cap_room_tokens) override {
        int best = best_of_k_near(rng_, ideal_tokens, cap_room_tokens, k_);
        if (best < 0) return PulledRequest{-1, 0};
        return PulledRequest{best, horizon_};  // prompt=실 길이, dtot=작은 고정 horizon.
    }
private:
    std::mt19937 rng_;
    int horizon_;
    int k_;

public:
    // best-of-K near-ideal (WorkloadSource::pull_near 와 동일 로직 — 재사용용 static).
    static int best_of_k_near(std::mt19937& rng, double ideal, long long cap, int k) {
        int best = -1;
        double best_dev = 1e300;
        for (int i = 0; i < k; ++i) {
            int length = sample_distribution_b(rng);
            if ((long long)length > cap) continue;
            double dev = std::fabs((double)length - ideal);
            if (dev < best_dev) { best_dev = dev; best = length; }
        }
        return best;
    }
};

// ── 2-페이즈 source: cold-fill 은 분포 B 자연 꼬리(long ~10%)로 풀을 seed, 그 후 round 루프는
//    best-of-K near-ideal(=ideal=hole 힐링이 like-for-like 로 작동). 한 source, 두 페이즈. ──
// 이유: 센터링 admit 의 ideal≈100K 는 best-of-K 로 long(≥256K)을 사실상 배제한다(check 3 1차에서
//   toxic_early=0 관측). 그래서 toxic 잔존을 논하려면 풀에 toxic 이 *실재*해야 한다. cold-fill 을
//   자연 분포로 seed 한 뒤, 완료-힐링 경로(advance_round → pull_near(ideal=hole))가 그 꼬리를
//   보존하는지 본다. heal 페이즈에서 best-of-K near-ideal 이므로 긴 hole→긴 교체 = toxic-fit.
class PhasedSource : public RequestSource {
public:
    PhasedSource(unsigned seed, int dec_horizon, int best_of_k = 200)
        : rng_(seed), horizon_(dec_horizon), k_(best_of_k), seeding_(true) {}
    void end_seeding() { seeding_ = false; }
    PulledRequest pull_near(double ideal_tokens, long long cap_room_tokens) override {
        int length;
        if (seeding_) {
            length = sample_distribution_b(rng_);              // 자연 꼬리 seed (ideal 무시).
            if ((long long)length > cap_room_tokens) length = (int)cap_room_tokens;
        } else {
            length = SmallDtotSource::best_of_k_near(rng_, ideal_tokens, cap_room_tokens, k_);
            if (length < 0) return PulledRequest{-1, 0};        // heal: ideal=hole 최근접.
        }
        return PulledRequest{length, horizon_};
    }
private:
    std::mt19937 rng_;
    int horizon_;
    int k_;
    bool seeding_;
};

static int count_toxic(const std::vector<NodeDecoder>& pool, int thresh) {
    int c = 0;
    for (const auto& q : pool) if (q.prompt >= thresh) ++c;
    return c;
}

// node_scheduler.cpp 내부 node_cap_room 과 동일식(거기 static 이라 재선언) — 대조 시뮬 cap 용.
static long long node_cap_room_proxy(const OperatingPoint& op) {
    return (long long)op.decode_pool * (long long)op.ctx_balance * 4;
}

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};
    OperatingPoint op = derive_operating_point(llama, b200, 128);
    std::printf("op: ctx_balance=%.0f decode_pool=%d decode_count_target=%d kv_target=%lld age_cap=%d\n",
                op.ctx_balance, op.decode_pool, op.decode_count_target,
                op.kv_operating_target, op.age_cap);

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 1 — 센터링 admit 수렴. decode_pool 회 admit_centered(true) → mean → ctx_balance(±10%).
    // ─────────────────────────────────────────────────────────────────────────
    {
        WorkloadSource src(/*seed*/12345);
        NodeScheduler node(op, src);
        for (int i = 0; i < op.decode_pool; ++i) node.admit_centered();

        double mean = node.mean_live_kv();
        std::printf("[1] centering: pool=%zu mean_live_kv=%.0f (ctx_balance=%.0f, ratio=%.3f)\n",
                    node.pool().size(), mean, op.ctx_balance, mean / op.ctx_balance);
        CHECK((int)node.pool().size() == op.decode_pool, "centering: pool filled to decode_pool");
        CHECK_REL(mean, op.ctx_balance, 0.10, "centering: mean_live_kv within +/-10% of ctx_balance");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 2 — 풀 유지. cold-fill 후 advance_round 여러 라운드 → pool().size() 가 decode_pool 근처.
    // (완료분만 hole→되채움. WorkloadSource 는 완료 드물어 풀이 떨어지지 않는 게 정상 동작.)
    // ─────────────────────────────────────────────────────────────────────────
    {
        WorkloadSource src(/*seed*/777);
        NodeScheduler node(op, src);
        for (int i = 0; i < op.decode_pool; ++i) node.admit_centered();
        int filled = (int)node.pool().size();

        int total_completions = 0;
        for (int r = 0; r < 60; ++r) total_completions += node.advance_round();

        int sz = (int)node.pool().size();
        std::printf("[2] pool-maintain: filled=%d after 60 rounds size=%d completions=%d\n",
                    filled, sz, total_completions);
        // per-completion 힐링은 hole 당 1 admit 으로 풀 크기 보존 → decode_pool 근처(±10%).
        CHECK(std::abs(sz - op.decode_pool) <= op.decode_pool / 10,
              "pool-maintain: size stays within 10% of decode_pool after rounds");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 3 — per-completion 힐링 toxic-fit 보존. PhasedSource 로 풀을 자연 분포(toxic 실재)로
    //   seed 한 뒤, 실제 node 코드 경로(advance_round → pull_near(ideal=hole))로 완료 churn 을
    //   돌려 상주 긴 요청(prompt≥256K) 비율 초기 vs 후기 비교 → 보존(급감 없음).
    //   대조군: 동일 seeded 풀에 batched(ideal=평균) 힐링을 적용한 순수 시뮬 — toxic 굶어 급감.
    // ─────────────────────────────────────────────────────────────────────────
    double frac_early = 0.0, frac_late = 0.0, frac_batched = 0.0;
    {
        const int TOXIC = 256000;     // 분포 B long 구간 하한.
        const int HORIZON = 8;        // dtot=horizon (완료가 수 라운드 안에 나도록).

        PhasedSource src(/*seed*/2024, HORIZON);
        NodeScheduler node(op, src);
        for (int i = 0; i < op.decode_pool; ++i) node.admit_centered(); // 자연 분포 seed.
        src.end_seeding();            // 이후 admit/heal 은 best-of-K near-ideal(=ideal=hole).

        // 초기 풀 스냅샷(대조군 시뮬과 공유) + early toxic.
        std::vector<NodeDecoder> seeded = node.pool();
        int toxic_early = count_toxic(seeded, TOXIC);
        frac_early = (double)toxic_early / seeded.size();

        int total_completions = 0;
        for (int r = 0; r < 200; ++r) total_completions += node.advance_round();
        int toxic_late = count_toxic(node.pool(), TOXIC);
        frac_late = (double)toxic_late / std::max<std::size_t>(node.pool().size(), 1);

        // ── 대조군: 동일 seeded 풀, batched(ideal=평균) 힐링. node_scheduler 미수정 — 순수 시뮬. ──
        // advance_round 와 동치 진행(라운드당 dec+1: 단순화해 모두 진행)으로 완료시 hole 을
        // *풀 평균* ideal 로 메운다(ARCHITECTURE §7.4 batched 대조). 긴 hole 도 평균 길이로 교체됨.
        {
            std::mt19937 brng(99);
            std::vector<NodeDecoder> bp = seeded;
            for (int r = 0; r < 200; ++r) {
                for (auto& q : bp) ++q.dec;               // 라운드당 진행(완료 유도 — 동형 churn).
                long long s = 0; for (auto& q : bp) s += q.live_kv();
                double mean = bp.empty() ? op.ctx_balance : (double)s / bp.size();
                std::vector<NodeDecoder> keep; int holes = 0;
                for (auto& q : bp) { if (q.dec >= q.dtot) ++holes; else keep.push_back(q); }
                bp.swap(keep);
                for (int h = 0; h < holes; ++h) {          // batched: ideal=평균(hole 무시).
                    int len = SmallDtotSource::best_of_k_near(brng, mean, node_cap_room_proxy(op), 200);
                    if (len < 0) continue;
                    bp.push_back(NodeDecoder{len, HORIZON, 0, 0});
                }
            }
            int toxic_b = count_toxic(bp, TOXIC);
            frac_batched = (double)toxic_b / std::max<std::size_t>(bp.size(), 1);
        }

        std::printf("[3] toxic-fit: completions=%d pool=%zu | per-completion early=%.3f late=%.3f"
                    " | batched-contrast late=%.3f\n",
                    total_completions, node.pool().size(), frac_early, frac_late, frac_batched);

        CHECK(total_completions > 0, "toxic-fit: synthetic source induced completions (churn observed)");
        CHECK(frac_early > 0.0, "toxic-fit: seeded pool initially contains toxic (long) requests");
        // per-completion(ideal=hole): 긴 hole 긴 교체 → 비율 보존(후기 ≥ 초기의 절반).
        CHECK(frac_late >= 0.5 * frac_early,
              "toxic-fit: long-request fraction PRESERVED under per-completion (ideal=hole) healing");
        // 대조: batched(ideal=평균) 는 같은 풀에서 toxic 을 굶긴다 → per-completion 이 엄밀히 우월.
        CHECK(frac_late > frac_batched,
              "toxic-fit: per-completion retains MORE toxic than batched (mean) healing (contrast)");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 4 — drift 없음. warmup 후 mean_live_kv 가 라운드 지나도 ctx_balance 근처 안정(early≈late).
    // ─────────────────────────────────────────────────────────────────────────
    {
        WorkloadSource src(/*seed*/555);
        NodeScheduler node(op, src);
        for (int i = 0; i < op.decode_pool; ++i) node.admit_centered();
        for (int r = 0; r < 10; ++r) node.advance_round();   // warmup

        double mean_early = node.mean_live_kv();
        for (int r = 0; r < 80; ++r) node.advance_round();   // 더 진행
        double mean_late = node.mean_live_kv();

        std::printf("[4] drift: mean_early=%.0f mean_late=%.0f ctx_balance=%.0f (early/ctx=%.3f late/ctx=%.3f)\n",
                    mean_early, mean_late, op.ctx_balance,
                    mean_early / op.ctx_balance, mean_late / op.ctx_balance);
        CHECK_REL(mean_late, op.ctx_balance, 0.15, "drift: mean_late near ctx_balance (no drift)");
        // early≈late: 두 측정이 서로 ±15% 안 — 라운드 지나도 표류하지 않음.
        CHECK(std::fabs(mean_late - mean_early) <= 0.15 * mean_early,
              "drift: early ~= late (stable, no creeping drift)");
    }

    return puls_test::summary();
}
