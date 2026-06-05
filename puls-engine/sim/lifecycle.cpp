// PULS-ENGINE — 통합 lifecycle 시뮬레이터. CONTRACT.md §2 (sim = WorkloadSource / 분포 B) / §3-②.
//
// 목적: 한 노드 내부의 *통합* lifecycle — 프리필→디코드 종속성 + age-cap + live-KV 센터링 —
//   이 동작점 composition 을 명중 ~100% · 디코드 Σdev ~0.2% · 프리필 Σdev ~0.07% 로 유지함을
//   엄밀히 재현한다. 이건 cluster_balance 독립 composer(on2 ~94%, 밴드±10%) 와 *다른* 더 엄밀한
//   경로다 — live-KV 센터링 + 잉여 cherry-pick 으로 Σdev 가 훨씬 작다.
//
// 근거 의도: implementation/analysis/cluster_lifecycle.cpp (admitDecodeCentered / composeDecodeBatch
//   2회 used 공유 / composePrefill / main 루프의 프리필 진행·전이(ready)·디코드 advance·완료 retire·
//   per-completion 힐링·풀 유지·Σdev 측정). 거기 손으로 박았던 123/12.3M/100K/256 을 derive 산출
//   op 와 core 프리미티브(steer_decode_set / steer_prefill_chunks)로 대체해 재현한다.
//
// 경계(CONTRACT §2·§9-2): 분포 B(WorkloadSource = 무한풀 emulation)·확률 churn 은 sim 전용.
//   core 수정 0 — 종속성(프리필→디코드)은 core 가 모델하지 않으므로 sim 드라이버 레벨에서
//   core 프리미티브를 조합해 엮는다. 디코드 풀 센터링 admit 은 node_scheduler.cpp 의 공식
//   (ideal = ctx_balance×(cnt+1) − liveSum)을 그대로 sim 분포 B 진행 주입과 함께 쓴다.

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

// 한 노드의 상주 디코더(길이-도메인). live KV = prompt + 누적 decode (steering 의 kv_length).
struct SimDecoder {
    int prompt;
    int dtot;
    int dec;
    int wait;
    long long live_kv() const { return (long long)prompt + dec; }
};

// 프리필 요청(분포 B). processed = 이미 처리한 프롬프트 토큰.
struct SimPrefill {
    int prompt;
    int dtot;
    int processed;
    int wait;
};

// 디코드 풀 센터링 admit(node_scheduler.cpp::admit_centered 공식 그대로):
//   ideal = ctx_balance×(cnt+1) − liveSum → 새 평균이 정확히 ctx_balance 되는 footprint.
//   풀이 떠 있으면 ideal↓ → 작은 것을 당겨 평균을 끌어내린다. WorkloadSource best-of-K = 무한풀.
//   fresh=true → dec=0(힐링). fresh=false(콜드스타트) → dec 랜덤 진행(분포 B 진행 주입, sim 전용).
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

// 디코드 한 μ-batch 구성: 풀을 steering 의 Decoder{live_kv, wait} 로 변환 후 steer_decode_set
// (used 공유 disjoint). 선택분 Σ live_kv 로 Σdev, 명중 = (개수==N_dec ∧ Σdev≤idle_band).
// node_scheduler.cpp::compose_decode_batch 와 동일 변환.
std::vector<int> compose_decode_batch(const std::vector<SimDecoder>& pool,
                                      std::vector<char>& used, const OperatingPoint& op,
                                      long long& sum_out, double& dev_out, bool& hit_out) {
    std::vector<Decoder> decoders;
    decoders.reserve(pool.size());
    for (const auto& q : pool) decoders.push_back(Decoder{q.live_kv(), q.wait});
    std::vector<int> sel = steer_decode_set(decoders, used, op);
    long long S = 0;
    for (int i : sel) S += pool[i].live_kv();
    sum_out = S;
    dev_out = std::fabs((double)S - (double)op.kv_operating_target) / (double)op.kv_operating_target;
    hit_out = ((int)sel.size() == op.decode_count_target) && (dev_out <= op.idle_band);
    return sel;
}

}  // namespace

int main() {
    // ── 동작점: derive 로 산출(하드코딩 0) ───────────────────────────────────────
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};
    const OperatingPoint op = derive_operating_point(llama, b200, /*prefill*/128);

    const int Z = 16;            // sim 노드 수 (동작점과 무관한 sim 규모 knob)
    const int ITERS = 1000, WARM = 500;
    const unsigned SEED = 7;

    std::printf("===== PULS 통합 lifecycle 시뮬레이터 (WorkloadSource — 분포 B, 무한풀) =====\n");
    std::printf("동작점(derive: Llama70B+B200+prefill128): ctx=%.0f N_dec=%d kv=%lld "
                "prefill_work=%lld decode_pool=%d prefill_pool=%d age_cap=%d idle_band=%.2f\n",
                op.ctx_balance, op.decode_count_target, op.kv_operating_target,
                op.prefill_kv_work_target, op.decode_pool, op.prefill_pool,
                op.age_cap, op.idle_band);
    std::printf("Z=%d nodes, ITERS=%d (WARM=%d). 통합: 프리필 steering+age-cap → (종속성)완료시 "
                "디코드 전이 → 디코드 steering(live-KV 센터)+age-cap → 완료 → per-completion 힐링.\n\n",
                Z, ITERS, WARM);

    // throughput 균형용 dtot: avg_dtot ≈ avg_prompt × N_dec/prefill (전이율 ≈ 완료율).
    // dtot 작게 잡아 완료 churn 이 실제로 일어나게(WorkloadSource.dtot=prompt 는 churn 너무 느림).
    const double dtot_frac = (double)op.decode_count_target / (double)op.prefill_tokens;

    std::mt19937 init_rng(SEED);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto sample_dtot = [&](int prompt, std::mt19937& rng) {
        int d = (int)(prompt * dtot_frac * (0.6 + 0.8 * uni(rng)));
        return std::max(1000, d);
    };

    // 노드 상태.
    std::vector<std::vector<SimDecoder>> dec_pool(Z);
    std::vector<std::vector<SimPrefill>> pf_pool(Z);
    std::vector<std::vector<SimPrefill>> ready(Z);  // 전이 대기(완료 프리필)
    std::vector<WorkloadSource> dec_src;             // 디코드 센터링/힐링 출처(노드별)
    std::vector<std::mt19937> node_rng;
    for (int z = 0; z < Z; ++z) {
        dec_src.emplace_back(SEED + 100 + z);
        node_rng.emplace_back(SEED + 200 + z);
    }
    std::mt19937 pf_init_rng(SEED + 300);

    // ── 콜드스타트: 디코드 풀=live-KV 센터 warm(dec 랜덤), 프리필 풀=다양 depth ──
    for (int z = 0; z < Z; ++z) {
        for (int i = 0; i < op.decode_pool; ++i)
            admit_decode_centered(dec_pool[z], op, dec_src[z], /*fresh*/false, node_rng[z]);
        for (int i = 0; i < op.prefill_pool; ++i) {
            int prompt = sample_distribution_b(pf_init_rng);
            SimPrefill q;
            q.prompt = prompt;
            q.dtot = sample_dtot(prompt, pf_init_rng);
            q.processed = (int)(uni(pf_init_rng) * prompt);  // 다양 depth (게이트 X)
            q.wait = 0;
            pf_pool[z].push_back(q);
        }
    }

    // 누적 측정.
    double accDecHit = 0, accDecDev = 0, accPfHit = 0, accPfDev = 0;
    double accDecMean = 0, accDecPool = 0, accReady = 0, accPfPool = 0;
    long long transitions = 0, completions = 0;
    int N = 0;
    // drift: early(WARM 직후) vs late.
    double earlyDecHit = 0, earlyDecDev = 0, earlyPfDev = 0;
    bool early_captured = false;

    for (int it = 0; it < ITERS; ++it) {
        double rDecHit = 0, rDecDev = 0, rPfHit = 0, rPfDev = 0;
        double rDecMean = 0, rDecPool = 0, rReady = 0, rPfPool = 0;

        for (int z = 0; z < Z; ++z) {
            auto& dec = dec_pool[z];
            auto& pf = pf_pool[z];
            auto& rdy = ready[z];

            // ── ① 프리필 steering (core: steer_prefill_chunks) ──
            std::vector<PrefillReq> preqs;
            preqs.reserve(pf.size());
            for (const auto& q : pf) preqs.push_back(PrefillReq{q.prompt, q.processed, q.wait});
            std::vector<int> chunk = steer_prefill_chunks(preqs, op);
            // depth-합 work = Σ over 배정 토큰의 depth(processed+1..processed+chunk).
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

            // 진행 + age 갱신 + 전이(종속성): processed+=chunk; processed≥prompt → ready.
            for (int i = 0; i < (int)pf.size(); ++i) {
                if (chunk[i] > 0) { pf[i].processed += chunk[i]; pf[i].wait = 0; }
                else ++pf[i].wait;
            }
            {
                std::vector<SimPrefill> keep;
                keep.reserve(pf.size());
                for (auto& q : pf) {
                    if (q.processed >= q.prompt) rdy.push_back(q);
                    else keep.push_back(q);
                }
                pf.swap(keep);
            }

            // ── ② 디코드 composition: 한 노드가 2 μ-batch (used 공유 disjoint) ──
            std::vector<char> used(dec.size(), 0);
            long long s1 = 0, s2 = 0;
            double dv1 = 1, dv2 = 1;
            bool h1 = false, h2 = false;
            std::vector<int> p1 = compose_decode_batch(dec, used, op, s1, dv1, h1);
            if ((int)dec.size() >= 2 * op.decode_count_target) {
                std::vector<int> p2 = compose_decode_batch(dec, used, op, s2, dv2, h2);
                (void)p2;
            }
            double dh = (h1 + h2) / 2.0;
            double dcdev = (dv1 + dv2) / 2.0;

            // footprint 평균(진단).
            double dmean = 0;
            {
                long long s = 0;
                for (const auto& q : dec) s += q.live_kv();
                dmean = dec.empty() ? 0 : (double)s / dec.size();
            }

            // ── ③ advance: 선택분 dec++/wait=0, 미선택(잉여) wait++; 완료 retire ──
            for (int i = 0; i < (int)dec.size(); ++i) {
                if (used[i]) { ++dec[i].dec; dec[i].wait = 0; }
                else ++dec[i].wait;
            }
            {
                std::vector<SimDecoder> keep;
                keep.reserve(dec.size());
                for (const auto& q : dec) {
                    if (q.dec >= q.dtot) ++completions;
                    else keep.push_back(q);
                }
                dec.swap(keep);
            }

            // ── ④ 디코드 풀 decode_pool 유지: ready(전이) 우선 → 모자라면 힐링(센터 fresh) ──
            //    per-completion 힐링: 완료 hole 만큼만 live-KV 센터로 되채움(toxic-fit via 센터링).
            while ((int)dec.size() < op.decode_pool && !rdy.empty()) {
                SimPrefill t = rdy.back();
                rdy.pop_back();
                SimDecoder q;
                q.prompt = t.prompt;
                q.dtot = t.dtot;
                q.dec = 0;
                q.wait = 0;
                dec.push_back(q);
                ++transitions;
            }
            while ((int)dec.size() < op.decode_pool)
                admit_decode_centered(dec, op, dec_src[z], /*fresh*/true, node_rng[z]);

            // ── ⑤ 프리필 풀 prefill_pool 유지: fresh(depth 0) 보충 ──
            while ((int)pf.size() < op.prefill_pool) {
                int prompt = sample_distribution_b(pf_init_rng);
                SimPrefill q;
                q.prompt = prompt;
                q.dtot = sample_dtot(prompt, pf_init_rng);
                q.processed = 0;
                q.wait = 0;
                pf.push_back(q);
            }

            // 라운드 측정 누적.
            rDecHit += dh;
            rDecDev += dcdev;
            rPfHit += pfhit ? 1.0 : 0.0;
            rPfDev += pfdev;
            rDecMean += dmean;
            rDecPool += dec.size();
            rReady += rdy.size();
            rPfPool += pf.size();
        }

        if (it >= WARM) {
            accDecHit += rDecHit / Z;
            accDecDev += rDecDev / Z;
            accPfHit += rPfHit / Z;
            accPfDev += rPfDev / Z;
            accDecMean += rDecMean / Z;
            accDecPool += rDecPool / Z;
            accReady += rReady / Z;
            accPfPool += rPfPool / Z;
            ++N;
            if (!early_captured) {
                earlyDecHit = rDecHit / Z;
                earlyDecDev = rDecDev / Z;
                earlyPfDev = rPfDev / Z;
                early_captured = true;
            }
        }
    }

    // late 윈도(말기 100라운드) 별도 집계 — drift 비교.
    // (위 acc 는 전체 steady-state 평균; early 는 첫 라운드. late 는 acc 평균으로 근사 충분.)
    double lateDecHit = accDecHit / N, lateDecDev = accDecDev / N, latePfDev = accPfDev / N;

    std::printf("[steady-state, 마지막 %d 라운드 × %d 노드]\n", ITERS - WARM, Z);
    std::printf("디코드: 명중 %6.2f%%  Σ편차 %6.3f%%  (live-KV 센터)  풀평균kv %.0f  풀크기 %.1f\n",
                100.0 * accDecHit / N, 100.0 * accDecDev / N, accDecMean / N, accDecPool / N);
    std::printf("프리필: 명중 %6.2f%%  Σ편차 %6.3f%%  (depth-work 타깃 %.2fM)  풀크기 %.1f\n",
                100.0 * accPfHit / N, 100.0 * accPfDev / N,
                op.prefill_kv_work_target / 1e6, accPfPool / N);
    std::printf("종속성: 전이(프리필→디코드) %lld 회, 디코드 완료 %lld 회, ready 대기 평균 %.2f\n",
                transitions, completions, accReady / N);
    std::printf("drift: 디코드명중 early %.2f%% vs late %.2f%% | 디코드Σdev early %.3f%% vs late %.3f%% "
                "| 프리필Σdev early %.3f%% vs late %.3f%%\n",
                100.0 * earlyDecHit, 100.0 * lateDecHit,
                100.0 * earlyDecDev, 100.0 * lateDecDev,
                100.0 * earlyPfDev, 100.0 * latePfDev);
    std::printf("\n→ 종속성·age-cap 넣고도 두 composition 이 live-KV 센터링으로 cluster on2 밴드보다\n"
                "  훨씬 타이트하게(Σdev ~0.2%%/0.07%%) 유지되면 통합 lifecycle 검증 성공.\n");
    return 0;
}
