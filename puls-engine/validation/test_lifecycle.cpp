// 검증 — 통합 lifecycle. CONTRACT.md §7 (통합). 엄밀 검증(CHECK).
//
// 목표: 노드 내부 통합 lifecycle(프리필→디코드 종속성 + age-cap + live-KV 센터링)이 동작점
//   composition 을 명중 ~100% · 디코드 Σdev ~0.2% · 프리필 Σdev ~0.07% 로 유지함을 엄밀히 재현.
//   이건 cluster_balance 독립 composer(on2 ~94%, 밴드±10%) 와 *다른* 더 엄밀한 경로다 —
//   live-KV 센터링 + 잉여 cherry-pick 으로 Σdev 가 훨씬 작다.
//
// 검증 항목:
//   - 디코드 명중률 ≥ 95%(목표 ~100%) AND 디코드 Σdev ≤ 0.5%(목표 ~0.2%).
//   - 프리필 명중 AND Σdev ≤ 0.2%(목표 ~0.07%).
//   - drift 없음(early≈late).
//   - 종속성: 전이(프리필→디코드) > 0, 디코드 완료 > 0.
//
// core 수정 0 — 종속성은 sim 드라이버 레벨에서 core 프리미티브(steer_decode_set /
//   steer_prefill_chunks)를 조합해 엮는다(node_scheduler.cpp 센터링 공식 재사용).

#include "test_framework.h"
#include "core/derive.h"
#include "core/operating_point.h"
#include "core/steering.h"
#include "core/workload.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace puls;

namespace {

struct SimDecoder {
    int prompt, dtot, dec, wait;
    long long live_kv() const { return (long long)prompt + dec; }
};
struct SimPrefill {
    int prompt, dtot, processed, wait;
};

// 디코드 풀 센터링 admit(node_scheduler.cpp::admit_centered 공식): ideal = ctx×(cnt+1) − liveSum.
void admit_decode_centered(std::vector<SimDecoder>& pool, const OperatingPoint& op,
                           WorkloadSource& src, bool fresh, std::mt19937& rng) {
    long long live_sum = 0;
    for (const auto& q : pool) live_sum += q.live_kv();
    int cnt = (int)pool.size();
    double ideal = op.ctx_balance * (cnt + 1) - (double)live_sum;
    if (ideal < 1000) ideal = 1000;
    long long cap_room = (long long)op.decode_pool * (long long)op.ctx_balance * 4;
    PulledRequest r = src.pull_near(ideal, cap_room);
    if (r.prompt < 0) return;
    SimDecoder q;
    q.prompt = r.prompt;
    q.dtot = r.dtot;
    q.dec = fresh ? 0 : (int)(std::uniform_real_distribution<double>(0.0, 1.0)(rng) * q.dtot);
    q.wait = 0;
    pool.push_back(q);
}

// 디코드 한 μ-batch 구성(core: steer_decode_set, used 공유). Σ live_kv·dev·hit 반환.
std::vector<int> compose_decode_batch(const std::vector<SimDecoder>& pool,
                                      std::vector<char>& used, const OperatingPoint& op,
                                      double& dev_out, bool& hit_out) {
    std::vector<Decoder> decoders;
    decoders.reserve(pool.size());
    for (const auto& q : pool) decoders.push_back(Decoder{q.live_kv(), q.wait});
    std::vector<int> sel = steer_decode_set(decoders, used, op);
    long long S = 0;
    for (int i : sel) S += pool[i].live_kv();
    dev_out = std::fabs((double)S - (double)op.kv_operating_target) / (double)op.kv_operating_target;
    hit_out = ((int)sel.size() == op.decode_count_target) && (dev_out <= op.idle_band);
    return sel;
}

// 통합 lifecycle 결과.
struct Result {
    double dec_hit;        // 디코드 명중률 (0..1)
    double dec_dev;        // 디코드 Σdev 평균 (0..1)
    double pf_hit;
    double pf_dev;
    double early_dec_hit, early_dec_dev, early_pf_dev;
    double late_dec_hit, late_dec_dev, late_pf_dev;
    long long transitions;
    long long completions;
};

// sim/lifecycle.cpp 와 동일 드라이버(검증용). 종속성·age-cap·센터링 통합 한 줄기.
Result run_lifecycle(const OperatingPoint& op, int Z, int ITERS, int WARM, unsigned SEED) {
    const double dtot_frac = (double)op.decode_count_target / (double)op.prefill_tokens;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::vector<SimDecoder>> dec_pool(Z);
    std::vector<std::vector<SimPrefill>> pf_pool(Z);
    std::vector<std::vector<SimPrefill>> ready(Z);
    std::vector<WorkloadSource> dec_src;
    std::vector<std::mt19937> node_rng;
    for (int z = 0; z < Z; ++z) {
        dec_src.emplace_back(SEED + 100 + z);
        node_rng.emplace_back(SEED + 200 + z);
    }
    std::mt19937 pf_rng(SEED + 300);
    auto sample_dtot = [&](int prompt, std::mt19937& rng) {
        return std::max(1000, (int)(prompt * dtot_frac * (0.6 + 0.8 * uni(rng))));
    };

    // 콜드스타트.
    for (int z = 0; z < Z; ++z) {
        for (int i = 0; i < op.decode_pool; ++i)
            admit_decode_centered(dec_pool[z], op, dec_src[z], false, node_rng[z]);
        for (int i = 0; i < op.prefill_pool; ++i) {
            int prompt = sample_distribution_b(pf_rng);
            pf_pool[z].push_back(SimPrefill{prompt, sample_dtot(prompt, pf_rng),
                                            (int)(uni(pf_rng) * prompt), 0});
        }
    }

    Result R{};
    double accDecHit = 0, accDecDev = 0, accPfHit = 0, accPfDev = 0;
    int N = 0;
    bool early_done = false;

    for (int it = 0; it < ITERS; ++it) {
        double rDecHit = 0, rDecDev = 0, rPfHit = 0, rPfDev = 0;
        for (int z = 0; z < Z; ++z) {
            auto& dec = dec_pool[z];
            auto& pf = pf_pool[z];
            auto& rdy = ready[z];

            // ① 프리필 steering.
            std::vector<PrefillReq> preqs;
            preqs.reserve(pf.size());
            for (const auto& q : pf) preqs.push_back(PrefillReq{q.prompt, q.processed, q.wait});
            std::vector<int> chunk = steer_prefill_chunks(preqs, op);
            double W = 0;
            int T = 0;
            for (int i = 0; i < (int)pf.size(); ++i) {
                if (chunk[i] > 0) {
                    int p0 = pf[i].processed;
                    W += (double)chunk[i] * p0 + (double)chunk[i] * (chunk[i] + 1) / 2.0;
                    T += chunk[i];
                }
            }
            double pfdev = std::fabs(W - (double)op.prefill_kv_work_target)
                         / (double)op.prefill_kv_work_target;
            bool pfhit = (T == op.prefill_tokens) && (pfdev <= op.idle_band);

            for (int i = 0; i < (int)pf.size(); ++i) {
                if (chunk[i] > 0) { pf[i].processed += chunk[i]; pf[i].wait = 0; }
                else ++pf[i].wait;
            }
            {
                std::vector<SimPrefill> keep;
                for (auto& q : pf) {
                    if (q.processed >= q.prompt) rdy.push_back(q);
                    else keep.push_back(q);
                }
                pf.swap(keep);
            }

            // ② 디코드 composition (2 μ-batch, used 공유).
            std::vector<char> used(dec.size(), 0);
            double dv1 = 1, dv2 = 1;
            bool h1 = false, h2 = false;
            compose_decode_batch(dec, used, op, dv1, h1);
            if ((int)dec.size() >= 2 * op.decode_count_target)
                compose_decode_batch(dec, used, op, dv2, h2);
            double dh = (h1 + h2) / 2.0;
            double dcdev = (dv1 + dv2) / 2.0;

            // ③ advance + 완료 retire.
            for (int i = 0; i < (int)dec.size(); ++i) {
                if (used[i]) { ++dec[i].dec; dec[i].wait = 0; }
                else ++dec[i].wait;
            }
            {
                std::vector<SimDecoder> keep;
                for (const auto& q : dec) {
                    if (q.dec >= q.dtot) ++R.completions;
                    else keep.push_back(q);
                }
                dec.swap(keep);
            }

            // ④ 디코드 풀 유지: ready 전이 우선 → 힐링.
            while ((int)dec.size() < op.decode_pool && !rdy.empty()) {
                SimPrefill t = rdy.back();
                rdy.pop_back();
                dec.push_back(SimDecoder{t.prompt, t.dtot, 0, 0});
                ++R.transitions;
            }
            while ((int)dec.size() < op.decode_pool)
                admit_decode_centered(dec, op, dec_src[z], true, node_rng[z]);

            // ⑤ 프리필 풀 유지.
            while ((int)pf.size() < op.prefill_pool) {
                int prompt = sample_distribution_b(pf_rng);
                pf.push_back(SimPrefill{prompt, sample_dtot(prompt, pf_rng), 0, 0});
            }

            rDecHit += dh;
            rDecDev += dcdev;
            rPfHit += pfhit ? 1.0 : 0.0;
            rPfDev += pfdev;
        }

        if (it >= WARM) {
            accDecHit += rDecHit / Z;
            accDecDev += rDecDev / Z;
            accPfHit += rPfHit / Z;
            accPfDev += rPfDev / Z;
            ++N;
            if (!early_done) {
                R.early_dec_hit = rDecHit / Z;
                R.early_dec_dev = rDecDev / Z;
                R.early_pf_dev = rPfDev / Z;
                early_done = true;
            }
            R.late_dec_hit = rDecHit / Z;
            R.late_dec_dev = rDecDev / Z;
            R.late_pf_dev = rPfDev / Z;
        }
    }
    R.dec_hit = accDecHit / N;
    R.dec_dev = accDecDev / N;
    R.pf_hit = accPfHit / N;
    R.pf_dev = accPfDev / N;
    return R;
}

}  // namespace

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};
    const OperatingPoint op = derive_operating_point(llama, b200, 128);

    Result R = run_lifecycle(op, /*Z*/16, /*ITERS*/1000, /*WARM*/500, /*SEED*/7);

    std::printf("통합 lifecycle (Llama70B+B200+prefill128): ctx=%.0f N_dec=%d kv=%lld work=%lld\n",
                op.ctx_balance, op.decode_count_target, op.kv_operating_target,
                op.prefill_kv_work_target);
    std::printf("디코드: 명중 %.2f%% Σdev %.3f%% | 프리필: 명중 %.2f%% Σdev %.3f%%\n",
                100.0 * R.dec_hit, 100.0 * R.dec_dev, 100.0 * R.pf_hit, 100.0 * R.pf_dev);
    std::printf("종속성: 전이 %lld, 완료 %lld\n", R.transitions, R.completions);
    std::printf("drift: 디코드명중 early %.2f%% late %.2f%% | 디코드Σdev early %.3f%% late %.3f%% | "
                "프리필Σdev early %.3f%% late %.3f%%\n",
                100.0 * R.early_dec_hit, 100.0 * R.late_dec_hit,
                100.0 * R.early_dec_dev, 100.0 * R.late_dec_dev,
                100.0 * R.early_pf_dev, 100.0 * R.late_pf_dev);

    // ── 디코드: live-KV 센터링이 cluster on2 밴드(±10%)보다 훨씬 타이트 ──
    CHECK(R.dec_hit >= 0.95, "decode hit-rate >= 95% (target ~100%)");
    CHECK(R.dec_dev <= 0.005, "decode Sigma-dev <= 0.5% (target ~0.2%, far tighter than on2 band)");

    // ── 프리필: depth-work steering 이 ~0.07% ──
    CHECK(R.pf_hit >= 0.99, "prefill hit-rate ~100%");
    CHECK(R.pf_dev <= 0.002, "prefill Sigma-dev <= 0.2% (target ~0.07%)");

    // ── drift 없음(early≈late) ──
    CHECK(std::fabs(R.early_dec_hit - R.late_dec_hit) <= 0.05, "decode hit no drift (early~late)");
    CHECK(std::fabs(R.early_dec_dev - R.late_dec_dev) <= 0.005, "decode Sigma-dev no drift");
    CHECK(std::fabs(R.early_pf_dev - R.late_pf_dev) <= 0.002, "prefill Sigma-dev no drift");

    // ── 종속성: 전이·완료가 실제로 일어남 ──
    CHECK(R.transitions > 0, "prefill->decode transitions occurred (dependency wired)");
    CHECK(R.completions > 0, "decode completions occurred (churn happens)");

    return puls_test::summary();
}
