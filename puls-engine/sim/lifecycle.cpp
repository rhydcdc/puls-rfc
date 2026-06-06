// PULS-ENGINE — 통합 lifecycle 시뮬레이터. CONTRACT.md §2 (sim = WorkloadSource / 분포 B) / §3-②.
//
// 목적: 한 노드 내부의 *통합* lifecycle — 프리필→디코드 종속성 + age-cap + per-completion
//   ideal=hole 힐링 + 엣지 게이팅 — 이 동작점 composition 을 디코드 ≈99.5% / Σdev ≈1.7%
//   (배포, age_cap 5) · 프리필 100% / ≈0.1% 로 유지함을 재현한다.
//
// 근거 의도: implementation/analysis/cluster_lifecycle.cpp (admitDecodeCentered / composeDecodeBatch
//   2회 used 공유 / composePrefill / main 루프의 프리필 진행·전이(ready)·디코드 advance·완료 retire·
//   per-completion 힐링·풀 유지·Σdev 측정). 거기 손으로 박았던 123/12.3M/100K/256 을 derive 산출
//   op 와 core 프리미티브(steer_decode_set / steer_prefill_chunks)로 대체해 재현한다.
//
// 경계(CONTRACT §2·§9-2): 분포 B(WorkloadSource = 무한풀 emulation)·확률 churn 은 sim 전용.
//   core 수정 0 — 종속성(프리필→디코드)은 core 가 모델하지 않으므로 sim 드라이버 레벨에서
//   core 프리미티브를 조합해 엮는다.
//   콜드스타트 = global_scheduler.cpp::cold_start (interleave-greedy min|mean−ctx_balance|,
//   다양성 보존) 그대로 재사용. 힐링 = node_scheduler.cpp::advance_round 의 per-completion
//   ideal=hole (like-for-like / toxic-fit). 평균(센터링) 힐링은 CONTRACT §4 금지(긴 거 굶음).

#include "core/derive.h"
#include "core/global_scheduler.h"
#include "core/operating_point.h"
#include "core/steering.h"
#include "core/workload.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// 분포 B 길이 클래스: 0=short[1K,16K) / 1=mid[16K,256K) / 2=long[256K,1M]. 유입 비율 20/70/10.
inline int cls(int prompt) { return prompt < 16000 ? 0 : (prompt < 256000 ? 1 : 2); }

// per-completion 힐링 admit (node_scheduler.cpp::advance_round ④ 그대로):
//   완료 retire 한 hole(=떠난 디코더의 prompt 길이)마다 ideal=hole 로 like-for-like 1개 되채움
//   (toxic-fit). 평균 센터링이 아니라 hole 그대로 → 긴 게 떠나면 긴 걸 당겨 분포 보존.
//   새 디코더는 dec=0(갓 admit). cap_room 은 node_scheduler.cpp::node_cap_room 와 동일.
int admit_hole(std::vector<SimDecoder>& pool, const OperatingPoint& op,
               WorkloadSource& src, int hole) {
    const long long cap_room = (long long)op.decode_pool * (long long)op.ctx_balance * 4;
    PulledRequest r = src.pull_near((double)hole, cap_room);
    if (r.prompt < 0) return -1;
    SimDecoder q;
    q.prompt = r.prompt;
    q.dtot = r.dtot;
    q.dec = 0;
    q.wait = 0;
    pool.push_back(q);
    return r.prompt;
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

int main(int argc, char** argv) {
    // ── 동작점: derive 로 산출(하드코딩 0) ───────────────────────────────────────
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};
    OperatingPoint op = derive_operating_point(llama, b200, /*prefill*/128);  // 배포 동작점(고정)

    const int ITERS = argc > 1 ? std::atoi(argv[1]) : 2000, WARM = 500;
    const int Z = argc > 2 ? std::atoi(argv[2]) : 16;          // sim 노드 수
    if (argc > 3) op.age_cap = std::atoi(argv[3]);             // age-cap (배포 5; 스윕 override)
    const int best_of_k = argc > 4 ? std::atoi(argv[4]) : 200; // 글로벌 후보 풀 richness(무한풀 emulation, sim 전용)
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

    // 현실적 decode 길이 모델: 출력 토큰수는 프롬프트와 무관한 짧은 분포(real serving).
    //   → live_kv = prompt + dec ≈ prompt 로 유지(누적 폭주 없음), 긴 요청도 완료·churn.
    //   (옛 모델 dtot ∝ prompt 는 긴 요청이 영원히 안 끝나 live_kv 를 153K 로 부풀렸음.)
    std::mt19937 init_rng(SEED);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto sample_dtot = [&](int /*prompt*/, std::mt19937& rng) {
        return 256 + (int)(uni(rng) * 3840);  // uniform[256, 4096] 출력 토큰
    };

    // 엣지 게이팅 컷오프(global_scheduler.cpp::gate): 큰 표본을 gate 해 interior arrival 의
    //   길이 상한을 구한다. 긴 요청(>cutoff)은 edge 노드로 — 워크로드 원래 평균과 무관하게
    //   interior 풀 평균을 ctx_balance 로 직접 맞춘다(ARCHITECTURE §7.3 엣지 게이팅).
    int edge_cutoff;
    double edge_frac;  // 전체 트래픽 중 edge 노드로 빠지는 비율(gate 가 shed 한 분율).
    {
        std::vector<int> big;
        big.reserve(200000);
        std::mt19937 grng(SEED + 999);
        for (int i = 0; i < 200000; ++i) big.push_back(sample_distribution_b(grng));
        GateResult g0 = gate(big, op);
        edge_cutoff = g0.kept.empty() ? (int)op.ctx_balance
                                      : *std::max_element(g0.kept.begin(), g0.kept.end());
        edge_frac = big.empty() ? 0.0 : (double)g0.edged.size() / (double)big.size();
    }
    auto gated_sample = [&](std::mt19937& rng) {
        int p;
        do { p = sample_distribution_b(rng); } while (p > edge_cutoff);
        return p;
    };
    std::printf("엣지 게이팅: edge_cutoff=%d 토큰, 전체 트래픽의 %.2f%% → edge 노드 "
                "(interior 평균을 ctx_balance≈%.0f 로 직접 정렬)\n\n",
                edge_cutoff, 100.0 * edge_frac, op.ctx_balance);

    // 노드 상태.
    std::vector<std::vector<SimDecoder>> dec_pool(Z);
    std::vector<std::vector<SimPrefill>> pf_pool(Z);
    std::vector<std::vector<SimPrefill>> ready(Z);  // 전이 대기(완료 프리필)
    std::vector<WorkloadSource> dec_src;             // 디코드 센터링/힐링 출처(노드별)
    std::vector<std::mt19937> node_rng;
    for (int z = 0; z < Z; ++z) {
        dec_src.emplace_back(SEED + 100 + z, best_of_k);
        node_rng.emplace_back(SEED + 200 + z);
    }
    std::mt19937 pf_init_rng(SEED + 300);

    // class 분포 점검 카운터 (short/mid/long): admit(콜드+전이+힐링) · 완료 · 상주 평균.
    long long admit_cls[3] = {0, 0, 0}, comp_cls[3] = {0, 0, 0};
    double resid_cls[3] = {0, 0, 0};

    // ── 콜드스타트: 디코드 풀 = global_scheduler.cpp::cold_start (interleave-greedy,
    //    min|추가후 mean − ctx_balance|, can_fit 게이트) 로 다양한 길이를 Z 노드에 분배.
    //    per-node 센터링(평균 최근접 1개 당김 → all-mid 붕괴)이 아니라 글로벌 배치 →
    //    분포 B(20/70/10) 보존. 배치된 길이를 SimDecoder 로 변환(dtot/진행은 sim 전용 주입).
    {
        // Z × decode_pool 개를 채우도록 충분히 큰 draw 를 분포 B 에서 뽑아 배치.
        std::vector<ClusterNode> cnodes(Z);
        std::vector<int> draw;
        draw.reserve(Z * op.decode_pool * 2);
        for (int i = 0; i < Z * op.decode_pool * 2; ++i)
            draw.push_back(sample_distribution_b(init_rng));
        // 엣지 게이팅: 최장 요청 shed → kept 평균 ≤ ctx_balance+edge_band (edged = edge 노드行).
        GateResult gres = gate(draw, op);
        cold_start(gres.kept, cnodes, op, SEED + 50);
        // 배치 결과(노드별 lengths)를 SimDecoder 로: dtot = sample_dtot, dec = 0 (cold_start 센터 보존).
        for (int z = 0; z < Z; ++z) {
            int take = std::min((int)cnodes[z].lengths.size(), op.decode_pool);
            for (int i = 0; i < take; ++i) {
                int prompt = cnodes[z].lengths[i];
                SimDecoder q;
                q.prompt = prompt;
                q.dtot = sample_dtot(prompt, node_rng[z]);
                q.dec = 0;  // cold_start 가 prompt 를 ctx_balance 로 센터 — warm 주입으로 live_kv 오염 금지
                q.wait = 0;
                dec_pool[z].push_back(q);
                ++admit_cls[cls(prompt)];
            }
            // cold_start can_fit 게이트로 모자라면 ideal=hole 로 보충(빈 풀이면 hole=ctx_balance).
            while ((int)dec_pool[z].size() < op.decode_pool) {
                int hole = dec_pool[z].empty()
                               ? (int)op.ctx_balance
                               : dec_pool[z][node_rng[z]() % dec_pool[z].size()].prompt;
                int p = admit_hole(dec_pool[z], op, dec_src[z], hole);
                if (p < 0) break;
                ++admit_cls[cls(p)];
            }
        }
    }
    for (int z = 0; z < Z; ++z) {
        for (int i = 0; i < op.prefill_pool; ++i) {
            int prompt = gated_sample(pf_init_rng);
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
    double accAgedDec = 0, accAgedPf = 0;  // aged(wait≥age_cap) 비율 — 디코드 vs 프리필 비교.
    double accDecForced = 0, accDecSel = 0;  // 배치당 강제 개수 / 선택 개수(=62면 count 충족).
    long long transitions = 0, completions = 0;
    int N = 0;
    // drift: early(WARM 직후) vs late.
    double earlyDecHit = 0, earlyDecDev = 0, earlyPfDev = 0;
    bool early_captured = false;

    for (int it = 0; it < ITERS; ++it) {
        double rDecHit = 0, rDecDev = 0, rPfHit = 0, rPfDev = 0;
        double rDecMean = 0, rDecPool = 0, rReady = 0, rPfPool = 0;
        double rResid[3] = {0, 0, 0};
        double rAgedDec = 0, rAgedPf = 0;
        double rDecForced = 0, rDecSel = 0;

        for (int z = 0; z < Z; ++z) {
            auto& dec = dec_pool[z];
            auto& pf = pf_pool[z];
            auto& rdy = ready[z];

            // age-cap 진단: 현재 aged(wait≥age_cap) 비율 (디코드 vs 프리필 비교용).
            { int a = 0; for (const auto& q : pf) if (q.wait >= op.age_cap) ++a;
              rAgedPf += pf.empty() ? 0 : (double)a / pf.size(); }

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
            { int f = 0; for (int i : p1) if (dec[i].wait >= op.age_cap) ++f;
              rDecForced += f; rDecSel += (double)p1.size(); }
            if ((int)dec.size() >= 2 * op.decode_count_target) {
                std::vector<int> p2 = compose_decode_batch(dec, used, op, s2, dv2, h2);
                (void)p2;
            }
            double dh = (h1 + h2) / 2.0;
            double dcdev = (dv1 + dv2) / 2.0;

            // footprint 평균(진단) + class 상주 분포(steer 시점).
            double dmean = 0;
            int rc[3] = {0, 0, 0};
            {
                long long s = 0;
                for (const auto& q : dec) { s += q.live_kv(); ++rc[cls(q.prompt)]; }
                dmean = dec.empty() ? 0 : (double)s / dec.size();
            }
            for (int c = 0; c < 3; ++c) rResid[c] += dec.empty() ? 0 : (double)rc[c] / dec.size();
            { int a = 0; for (const auto& q : dec) if (q.wait >= op.age_cap) ++a;
              rAgedDec += dec.empty() ? 0 : (double)a / dec.size(); }

            // ── ③ advance: 선택분 dec++/wait=0, 미선택(잉여) wait++; 완료 retire + hole 수집 ──
            //    node_scheduler.cpp::advance_round ③: dec≥dtot 완료 → hole(prompt) 모음.
            for (int i = 0; i < (int)dec.size(); ++i) {
                if (used[i]) { ++dec[i].dec; dec[i].wait = 0; }
                else ++dec[i].wait;
            }
            std::vector<int> holes;
            {
                std::vector<SimDecoder> keep;
                keep.reserve(dec.size());
                for (const auto& q : dec) {
                    if (q.dec >= q.dtot) {
                        ++completions;
                        ++comp_cls[cls(q.prompt)];
                        holes.push_back(q.prompt);  // ideal=hole 되채움용 (like-for-like)
                    } else {
                        keep.push_back(q);
                    }
                }
                dec.swap(keep);
            }

            // ── ④ per-completion 힐링: hole 마다 정확히 1개 되채움 (node_scheduler.cpp ④).
            //    완료한 hole 단위로 되채움 — ready(프리필 전이) 우선, 모자라면 ideal=hole admit.
            //    canonical(CONTRACT §4): batched(평균/센터링) 힐링 금지 — 긴 거 굶음.
            for (int hole : holes) {
                if (!rdy.empty()) {
                    // ready 전이 우선: 완료된 프리필을 디코드로 전이(종속성, sim 전용).
                    SimPrefill t = rdy.back();
                    rdy.pop_back();
                    SimDecoder q;
                    q.prompt = t.prompt;
                    q.dtot = t.dtot;
                    q.dec = 0;
                    q.wait = 0;
                    dec.push_back(q);
                    ++transitions;
                    ++admit_cls[cls(t.prompt)];
                } else {
                    // ideal = hole 로 like-for-like 되채움 (toxic-fit).
                    int p = admit_hole(dec, op, dec_src[z], hole);
                    if (p < 0) break;  // 소스 고갈 시 차단
                    ++admit_cls[cls(p)];
                }
            }
            // ready 가 hole 보다 많이 쌓였으면(전이 적체) 남는 슬롯까지 전이로 흡수.
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
                ++admit_cls[cls(t.prompt)];
            }

            // ── ⑤ 프리필 풀 prefill_pool 유지: fresh(depth 0) 보충 ──
            while ((int)pf.size() < op.prefill_pool) {
                int prompt = gated_sample(pf_init_rng);
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
            for (int c = 0; c < 3; ++c) resid_cls[c] += rResid[c] / Z;
            accAgedDec += rAgedDec / Z;
            accAgedPf += rAgedPf / Z;
            accDecForced += rDecForced / Z;
            accDecSel += rDecSel / Z;
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
    std::printf("class admit(누적) short/mid/long: %lld / %lld / %lld  |  완료: %lld / %lld / %lld\n",
                admit_cls[0], admit_cls[1], admit_cls[2], comp_cls[0], comp_cls[1], comp_cls[2]);
    std::printf("class 상주분포(post-warm 평균) short/mid/long: %.1f%% / %.1f%% / %.1f%%  (분포B 유입 = 20/70/10)\n",
                100.0 * resid_cls[0] / N, 100.0 * resid_cls[1] / N, 100.0 * resid_cls[2] / N);
    std::printf("\n[SUMMARY] decode_pool=%d age_cap=%d Z=%d bestK=%d | edge=%.2f%% cutoff=%d | "
                "DECODE hit=%.2f%% Sdev=%.3f%% mean=%.0f | PREFILL hit=%.2f%% Sdev=%.3f%% | "
                "aged d/p=%.1f/%.1f%% | decsel/forced=%.1f/%.1f | resid s/m/l=%.1f/%.1f/%.1f%% | trans=%lld comp=%lld\n",
                op.decode_pool, op.age_cap, Z, best_of_k, 100.0 * edge_frac, edge_cutoff,
                100.0 * accDecHit / N, 100.0 * accDecDev / N, accDecMean / N,
                100.0 * accPfHit / N, 100.0 * accPfDev / N,
                100.0 * accAgedDec / N, 100.0 * accAgedPf / N,
                accDecSel / N, accDecForced / N,
                100.0 * resid_cls[0] / N, 100.0 * resid_cls[1] / N, 100.0 * resid_cls[2] / N,
                transitions, completions);
    return 0;
}
