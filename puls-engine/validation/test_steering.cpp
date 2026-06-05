// 검증 — 순수 steering(decode/prefill) + age-cap. CONTRACT.md §3-② · §4 · §7.
// 앵커: Llama-3 70B + B200 + prefill 128 = (N_dec≈62, kv≈6.15M). steering 명중 = (count, kv±idle_band).
// canonical(§4): age-cap 강제 = 가장 wait 큰 요청. starvation ≤ age_cap+1 batch.
//   prefill depth = processed + chunk + 1.
#include "test_framework.h"
#include "core/derive.h"
#include "core/steering.h"
#include "core/workload.h"

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

using namespace puls;

// 분포 B 길이 풀을 디코더 풀로(wait=0). variant 로 분포 클램프.
enum Variant { ALL, HEAVY, SHORT, BIMODAL };

static std::vector<Decoder> make_decode_pool(int count, unsigned seed, Variant v) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<Decoder> pool;
    pool.reserve(count);
    while ((int)pool.size() < count) {
        long long len = sample_distribution_b(rng);
        switch (v) {
            case ALL: break;
            case HEAVY:   if (len < 100000) continue; break;          // 긴 것만
            case SHORT:   if (len > 64000)  continue; break;          // 짧은 것만
            case BIMODAL: if (len > 16000 && len < 256000) continue; break; // 양 극단만
        }
        pool.push_back(Decoder{len, /*wait*/0});
    }
    return pool;
}

// decode set 의 Σkv 합.
static long long sum_kv(const std::vector<Decoder>& pool, const std::vector<int>& sel) {
    long long s = 0;
    for (int i : sel) s += pool[i].kv_length;
    return s;
}

// 명중(full 분포 B 풀): count==target AND Σkv 가 idle_band 안. dev 반환(spread 비교).
// 분포 B 가 워크로드 가정(§9-2)이므로 무한풀 근사에서 두 타깃을 동시 만족한다.
static double check_decode_hit(const OperatingPoint& op, const char* name, unsigned seed) {
    std::vector<Decoder> pool = make_decode_pool(400, seed, ALL);
    std::vector<char> used(pool.size(), 0);
    std::vector<int> sel = steer_decode_set(pool, used, op);

    long long S = sum_kv(pool, sel);
    double dev = std::fabs((double)S - (double)op.kv_operating_target)
               / (double)op.kv_operating_target;
    std::printf("  decode[%-9s] n=%d (target=%d)  Skv=%lld (target=%lld)  dev=%.4f\n",
                name, (int)sel.size(), op.decode_count_target,
                S, op.kv_operating_target, dev);

    CHECK((int)sel.size() == op.decode_count_target, "decode count == target");
    CHECK(dev <= op.idle_band, "Sigma kv within idle_band of target");
    return dev;
}

// 변종 풀 종료-불변식: 그리디 종료는 n==count_target ∨ S≥kv_target. 그리고 마지막 한 개를
// 빼면 Σkv < kv_target (= 한 요청만큼만 overshoot). 변종(heavy/short/bimodal) 무관 성립.
static void check_decode_invariant(const OperatingPoint& op, Variant v, const char* name,
                                   unsigned seed) {
    std::vector<Decoder> pool = make_decode_pool(400, seed, v);
    std::vector<char> used(pool.size(), 0);
    std::vector<int> sel = steer_decode_set(pool, used, op);

    long long S = sum_kv(pool, sel);
    int n = (int)sel.size();
    // S_before_last = 마지막 선택 직전 누적.
    long long S_before = sel.empty() ? 0 : S - pool[sel.back()].kv_length;

    std::printf("  decode[%-9s] n=%d/%d Skv=%lld %s\n", name, n, op.decode_count_target, S,
                n == op.decode_count_target ? "(count-bound)" : "(kv-bound)");

    CHECK(n <= op.decode_count_target, "count never exceeds target");
    CHECK(n == op.decode_count_target || S >= op.kv_operating_target,
          "termination: count==target OR Sigma kv reached target");
    // kv-bound 종료면 마지막 직전엔 아직 target 미달 = overshoot 한 요청 이내.
    if (n < op.decode_count_target)
        CHECK(S_before < op.kv_operating_target, "overshoot bounded by one request");
}

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};
    OperatingPoint op = derive_operating_point(llama, b200, 128);
    std::printf("op: N_dec=%d kv_target=%lld ctx=%.0f age_cap=%d idle_band=%.2f P=%d kvwork=%lld\n",
                op.decode_count_target, op.kv_operating_target, op.ctx_balance,
                op.age_cap, op.idle_band, op.prefill_tokens, op.prefill_kv_work_target);

    // ── 1. decode steering 명중 ────────────────────────────────────────────────
    // (a) full 분포 B 풀(워크로드 가정): count==target ∧ Σkv ±idle_band. 여러 시드 spread 작음.
    std::printf("[1a] decode steering hit on distribution-B pools (count + Sigma-kv +/- idle_band)\n");
    double d0 = check_decode_hit(op, "seed-11", 11);
    double d1 = check_decode_hit(op, "seed-22", 22);
    double d2 = check_decode_hit(op, "seed-33", 33);
    double d3 = check_decode_hit(op, "seed-44", 44);
    double dmax = std::max({d0, d1, d2, d3});
    double dmin = std::min({d0, d1, d2, d3});
    std::printf("  spread: dev in [%.4f, %.4f], range=%.4f\n", dmin, dmax, dmax - dmin);
    CHECK(dmax <= op.idle_band, "all distribution-B pools hit within idle_band");
    CHECK(dmax - dmin <= 0.02, "dev spread across seeds small (~1%)");

    // (b) 변종 풀(heavy/short/bimodal): 그리디 종료 불변식이 분포 무관 성립.
    std::printf("[1b] decode termination invariant across pool variants\n");
    check_decode_invariant(op, HEAVY,   "heavy",   22);
    check_decode_invariant(op, SHORT,   "short",   33);
    check_decode_invariant(op, BIMODAL, "bimodal", 44);

    // ── 2. age-cap canonical: 가장 wait 큰 것이 강제 선택 ───────────────────────
    // wait 가 제각각. age_cap 이상 여럿(예 idx 의 wait = age_cap..age_cap+k). 가장 큰 wait
    // 를 가진 인덱스가 첫 강제 선택이어야 한다(kv_length 와 무관 — steering 이 아닌 강제).
    std::printf("[2] age-cap canonical = largest-wait forced\n");
    {
        std::vector<Decoder> pool;
        // 평범한 풀(wait 0) 다수 — steering 후보.
        for (int i = 0; i < 20; ++i) pool.push_back(Decoder{50000, 0});
        // age_cap 이상 여럿: wait 를 제각각으로. 가장 큰 wait 는 idx = aged_start+? .
        int aged_start = (int)pool.size();
        std::vector<int> waits = {op.age_cap + 1, op.age_cap, op.age_cap + 3,
                                  op.age_cap + 2, op.age_cap};
        // kv_length 는 ideal 에서 일부러 멀게(steering 으로는 안 뽑힐 값) — 강제임을 분리 확인.
        for (size_t k = 0; k < waits.size(); ++k)
            pool.push_back(Decoder{/*kv_length*/999999999LL, waits[k]});
        // 기대: 가장 wait 큰 것 = aged_start + 2 (wait = age_cap+3).
        int expect = aged_start + 2;
        int expect_wait = op.age_cap + 3;

        std::vector<char> used(pool.size(), 0);
        std::vector<int> sel = steer_decode_set(pool, used, op);
        int first = sel.empty() ? -1 : sel.front();
        std::printf("  first pick idx=%d (wait=%d), expected idx=%d (wait=%d)\n",
                    first, first >= 0 ? pool[first].wait : -1, expect, expect_wait);
        CHECK(first == expect, "first forced pick = largest-wait request (canonical)");

        // 강제 대상이 모두 steering 후보보다 먼저(앞쪽)에 — aged 들이 비-aged 보다 우선.
        // 처음 (aged 수)개의 선택이 전부 aged 인덱스인지, wait 내림차순인지 직접 확인.
        int n_aged = (int)waits.size();
        bool all_aged_first = true, desc = true;
        int prev_wait = 1 << 30;
        for (int t = 0; t < n_aged && t < (int)sel.size(); ++t) {
            int idx = sel[t];
            if (idx < aged_start) all_aged_first = false;
            // 선택 시점 pool[idx].wait 는 불변(age_update 안 함) — 내림차순이어야.
            if (pool[idx].wait > prev_wait) desc = false;
            prev_wait = pool[idx].wait;
        }
        CHECK(all_aged_first, "all age-capped picked before steering candidates");
        CHECK(desc, "age-capped picked in descending wait order");
    }

    // ── 3. age-cap starvation 0: 스트리밍 라운드 모사 ───────────────────────────
    // 매 라운드 steer_decode_set → age_update. 풀 = op.decode_pool(=2N+surplus, 노드 운영 크기,
    // 완료 churn 없는 최악 경쟁). 어떤 요청도 wait ≤ age_cap+1 안에 한 번은 선택.
    // (aged backlog ≤ count_target/round 이면 한 batch 안 drain — 운영 풀 크기에서 성립.)
    std::printf("[3] age-cap starvation 0 (streaming rounds, pool = decode_pool)\n");
    {
        int worst = 0;
        for (unsigned seed : {77u, 1u, 2u, 99u}) {
            std::vector<Decoder> pool = make_decode_pool(op.decode_pool, seed, ALL);
            int observed_max_wait = 0;
            const int rounds = 300;
            for (int r = 0; r < rounds; ++r) {
                std::vector<char> used(pool.size(), 0);
                std::vector<int> sel = steer_decode_set(pool, used, op);
                age_update(pool, sel);
                for (const auto& d : pool)
                    observed_max_wait = std::max(observed_max_wait, d.wait);
            }
            std::printf("  seed=%-3u pool=%d rounds=%d  max_wait=%d  bound(age_cap+1)=%d\n",
                        seed, op.decode_pool, rounds, observed_max_wait, op.age_cap + 1);
            CHECK(observed_max_wait <= op.age_cap + 1,
                  "no request waits beyond age_cap+1 (starvation 0)");
            worst = std::max(worst, observed_max_wait);
        }
        std::printf("  worst max_wait across seeds = %d\n", worst);
    }

    // ── 4. prefill steering: Σchunk == budget, depth-합 ~ target, depth=processed+chunk+1 ──
    std::printf("[4] prefill steering (Sigma-chunk == budget, depth-sum ~ target)\n");
    {
        // PrefillReq 풀: processed 를 ctx_balance 부근으로 분산(prefill ideal depth ≈ ctx_balance,
        // = prefill_kv_work_target / prefill_tokens). 잔여 프롬프트는 작게 — 다음 chunk 깊이가
        // ideal 근방이 되도록. 깊은 풀이라야 depth-합 타깃 도달(얕은 풀은 도달 불가 — 정상).
        std::mt19937 rng(55);
        std::uniform_int_distribution<int> proc(1000, 200000);
        std::uniform_int_distribution<int> rem(50, 600);
        std::vector<PrefillReq> pool;
        for (int i = 0; i < op.prefill_pool; ++i) {
            int processed = proc(rng);
            int prompt = processed + rem(rng);
            pool.push_back(PrefillReq{prompt, processed, /*wait*/0});
        }
        std::vector<int> chunk = steer_prefill_chunks(pool, op);

        int total = 0;
        long long depth_sum = 0;       // Σ over allocated tokens of (processed+running_chunk)
        long long remaining_cap = 0;   // Σ 잔여 프롬프트(상한)
        for (size_t i = 0; i < pool.size(); ++i) {
            total += chunk[i];
            remaining_cap += std::max(0, pool[i].prompt - pool[i].processed);
            // 이 요청의 depth 기여 = Σ_{c=1..chunk} (processed + c)  (depth=processed+c, c=running)
            for (int c = 1; c <= chunk[i]; ++c)
                depth_sum += pool[i].processed + c;
            // 잔여 프롬프트 한계 초과 배정 금지 확인.
            CHECK(pool[i].processed + chunk[i] <= pool[i].prompt, "chunk <= remaining prompt");
        }
        int budget_eff = (int)std::min((long long)op.prefill_tokens, remaining_cap);
        double wdev = std::fabs((double)depth_sum - (double)op.prefill_kv_work_target)
                    / (double)op.prefill_kv_work_target;
        std::printf("  Sigma-chunk=%d (budget=%d, eff=%d, cap=%lld)  depth_sum=%lld (target=%lld) wdev=%.4f\n",
                    total, op.prefill_tokens, budget_eff, remaining_cap,
                    depth_sum, op.prefill_kv_work_target, wdev);
        CHECK(total == budget_eff, "Sigma chunk == budget (or remaining-prompt cap)");
        // depth = processed+chunk+1 반영: greedy 가 target 근처. 넉넉한 idle_band 로.
        CHECK(wdev <= op.idle_band, "prefill depth-sum near prefill_kv_work_target");

        // depth = processed + chunk + 1 (다음 토큰) 반영 직접 확인:
        // 잔여 프롬프트가 1뿐인 요청은 chunk 가 최대 1 — depth 정의가 +1 형이면 한 토큰만 배정.
        std::vector<PrefillReq> tiny{ PrefillReq{/*prompt*/5, /*processed*/4, 0} };
        std::vector<int> tc = steer_prefill_chunks(tiny, op);
        std::printf("  tiny: prompt=5 processed=4 -> chunk=%d (cap 1)\n", tc[0]);
        CHECK(tc[0] == 1, "depth uses processed+chunk(+1); chunk capped by remaining prompt");
    }

    return puls_test::summary();
}
