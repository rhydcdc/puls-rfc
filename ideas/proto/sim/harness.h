// proto/sim — 공용 하니스 (A·B 비교 가능성 고정, header-only).
// 동작점 도출 · 엣지 게이팅 컷오프 · 멀티턴 복귀 스케줄 · 캐시 용량 · 시간(us) 모델.
// 두 드라이버(baseline.cpp / prepo.cpp)가 이 셋업을 공유해 같은 워크로드 위에서 대조된다.
#pragma once
#include <algorithm>
#include <vector>
#include <map>
#include <random>
#include "core/derive.h"
#include "core/optime.h"             // t_pim_us / t_ffn_us / t_gpu_a_us (TBT·idle-win)
#include "core/global_scheduler.h"   // gate
#include "core/workload.h"           // sample_distribution_b
#include "scheduler/request.h"
#include "sim/workload_mt.h"

namespace proto {

// 배포 인스턴스: Llama70B + B200 + prefill 128. (모델/HW 는 입력 스펙 — 동작점 수치는 derive.)
struct Deployment {
    puls::ModelSpec       model;
    puls::HwSpec          hw;
    puls::OperatingPoint  op;
};
inline Deployment make_deployment() {
    puls::ModelSpec m{80, 8192, 64, 8, 128, 28672};
    puls::HwSpec    hw{2200.0, 0.6};
    return Deployment{m, hw, puls::derive_operating_point(m, hw, 128)};
}

// ── 시간 모델 (단위 = µs) ─────────────────────────────────────────────────────
// forward-pass 1 라운드 = balance_time_us × num_layers (= X·L ≈ 2.0 ms 배포).
inline double round_us(const Deployment& d) {
    return d.op.balance_time_us * (double)d.model.num_layers;
}
// 프리필 시간: prefill_tokens 개 = 1 라운드. → len 토큰 ≈ round_us × len/prefill_tokens.
inline double prefill_us(const Deployment& d, long long len) {
    return round_us(d) * (double)len / (double)d.op.prefill_tokens;
}

// ── 라운드 op-time (TBT·idle-win) ─────────────────────────────────────────────
// 한 μ-batch 의 세 자원 시간(µs). decode_count·Σdecode_kv 는 실현값, prefill 항은 동작점 타깃.
struct OpTimes { double t_pim, t_gpu_a, t_ffn; };
inline OpTimes round_optimes(int decode_count, long long sum_decode_kv, const Deployment& d) {
    const int batch_total = decode_count + d.op.prefill_tokens;
    return OpTimes{
        puls::t_pim_us(sum_decode_kv, d.hw),
        puls::t_gpu_a_us(batch_total, d.op.prefill_kv_work_target, d.model, d.hw),
        puls::t_ffn_us(batch_total, d.model, d.hw)
    };
}

// 인스턴스 A 지연 (PIM ∥ GPU-A) + HBM 컨텐션 페널티. Plan3 §3.
//   컨텐션-무료 조건 = t_pim ≤ t_gpu_a (PIM 이 GPU-A 그림자에 숨음) → 페널티 0.
//   위반(t_pim > t_gpu_a, PIM 노출) → β · (노출분). max(0,·) 가 게이트라 β 켜도 음수 안 더해짐.
inline double instance_a_latency(const OpTimes& t, double beta) {
    const double exposed = t.t_pim - t.t_gpu_a;        // >0 이면 PIM 노출
    return std::max(t.t_pim, t.t_gpu_a) + beta * std::max(0.0, exposed);
}

// ── 엣지 게이팅 컷오프: 큰 표본을 gate 해 interior arrival 길이 상한 ──────────────
inline int edge_cutoff(const puls::OperatingPoint& op, unsigned seed) {
    std::vector<int> big;
    big.reserve(200000);
    std::mt19937 g(seed + 999);
    for (int i = 0; i < 200000; ++i) big.push_back(puls::sample_distribution_b(g));
    puls::GateResult gr = puls::gate(big, op);
    if (gr.kept.empty()) return (int)op.ctx_balance;
    int mx = 0;
    for (int x : gr.kept) if (x > mx) mx = x;
    return mx;
}

// ── 노드 캐시 용량(바이트) = derive 잔여 (HBM − Instance A) ────────────────────
// instance_a_tb 는 decode_pool(=2배치+잉여) 기준 → 이 값은 *잉여까지 점유*한 잔여(A·C 용).
inline long long node_cache_capacity_bytes(const puls::OperatingPoint& op) {
    double spare_tb = op.hbm_capacity_tb - op.instance_a_tb;
    if (spare_tb < 0) spare_tb = 0;
    return (long long)(spare_tb * 1e12);
}
// B 는 잉여가 없어(124 상주) 잉여 KV 만큼 캐시 용량이 더 크다. 그 추가분(바이트).
inline long long surplus_kv_bytes(const Deployment& d) {
    const int surplus = d.op.decode_pool - d.op.node_min;   // = decode_surplus (예 10)
    return (long long)((double)surplus * d.op.ctx_balance * (double)d.model.kv_bytes_per_token());
}

// ── 미래 라운드에 도착할 멀티턴 복귀들 ────────────────────────────────────────
struct PendingReturns {
    std::multimap<long long, Request> by_round;
    void add(long long round, const Request& r) { by_round.emplace(round, r); }
    std::vector<Request> due(long long now) {
        std::vector<Request> out;
        auto it = by_round.begin();
        while (it != by_round.end() && it->first <= now) {
            out.push_back(it->second);
            it = by_round.erase(it);
        }
        return out;
    }
};

// ── 공용 실험 설정 (A·B 동일) ─────────────────────────────────────────────────
struct SimConfig {
    int Z      = 64;       // 노드 수
    int iters  = 4000;
    int warm   = 500;
    unsigned seed = 7;
    // 인구 보존 모델: 도착 = 이탈(대화 종료 시 fresh 1, 복귀는 예약). flood 없음.
    // 초기 인구 = 노드 채움 + 큐 backlog. backlog 은 *절대값*(노드용량 배수 아님) — 회전율이
    // 낮아(디코더 장수명) backlog 이 크면 대기가 age_cap 을 넘겨 강제가 폭주하므로,
    // backlog ≈ 회전율(≈Z×pool/평균max_tokens) × age_cap 규모로 작게 둔다.
    int seed_backlog = 300;

    // 스윕 knob (3) + 고정 가정. 기본값은 회전율(~3.6/round)·backlog(300)와 정합하는 레짐:
    //   대기 ~ backlog/rate ~ 80 라운드 → age_cap·evict_age 를 그 위로.
    int    eligibility_threshold = 16000;   // HBM 캐싱 적격 최소 길이 (스윕). baseline 은 무한대.
    int    evict_age             = 200;     // HBM idle 상한 라운드 (스윕)
    int    gone_age              = 3000;    // SSD 소멸 라운드 (고정 가정)
    int    global_age_cap        = 2000;    // 글로벌 풀 대기 상한 라운드 (스윕) — Plan2 재튜닝: 100 은
                                            //   강제 61%로 과도해 Σdev↑. 2000(강제~소수)으로 상향, knee 는 §4 스윕.
    // 재로드 속도 라벨(바이트/라운드). Plan3 교정: SSD ~10 GB/s = 2e7 B/round (옛 5e9 = HBM급 오기).
    double offload_bw_bytes_per_round = 2.0e7;

    // HBM 컨텐션 β (Plan3 §3): A_time = max(pim,gpua) + β·max(0, pim−gpua). 0=무컨텐션. 스윕 knob.
    double contention_beta = 0.5;

    // SLO (실 KPI). 임계는 가정 라벨(스윕 가능). T_tbt 는 round_us 배수로 드라이버가 산출.
    double slo_ttft_us  = 5.0e6;   // TTFT SLO (≈5 s)
    double slo_tbt_mult = 1.30;    // TBT SLO = slo_tbt_mult × round_us(d)

    // 멀티턴 워크로드 (short<mid<long 복귀확률, 평균 ~2.5턴)
    MultiTurnConfig mt{
        /*p_return_short*/0.35, /*mid*/0.60, /*long*/0.75,
        /*dtot_lo*/256,   /*dtot_hi*/4096,
        /*new_msg_lo*/500,/*new_msg_hi*/8000,
        /*think_gap_lo*/2,/*think_gap_hi*/40
    };
};

} // namespace proto
