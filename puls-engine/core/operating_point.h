// PULS-ENGINE — 도출된 동작점. CONTRACT.md §5.2. derive 의 출력, 노드/글로벌의 입력.
#pragma once

namespace puls {

// 산출 모듈(derive)의 결과. 모든 수치는 (ModelSpec, HwSpec, prefill_tokens) 에서 도출 — 리터럴 0.
struct OperatingPoint {
    // 제어 타깃
    double    ctx_balance;            // 균형 ctx (tokens) — 문서의 ~100K
    int       decode_count_target;    // N_dec
    long long kv_operating_target;    // = N_dec × ctx_balance (Σ decode KV tokens)
    int       prefill_tokens;         // knob P
    long long prefill_kv_work_target; // = P × ctx_balance (Σ chunk×depth 타깃)
    int       ffn_batch;              // = N_dec + P
    double    balance_time_us;        // X (= t_PIM ≈ t_FFN ≈ t_GPU-A)

    // 풀 구성
    int decode_pool;                  // = 2×N_dec + decode_surplus
    int prefill_pool;                 // depth-diversity 하한 + 마진
    int age_cap;                      // 공정성 강제 임계

    // 진단(제어값 아님)
    double idle_band;                 // ±밴드 (0.10) — idle-SLA 라벨

    // 클러스터 라우팅 경계
    int node_max;                     // 노드당 decode 풀 *개수* 상한
    int node_min;                     // = 2×N_dec (2 μ-batch 바닥)
    double edge_band;                 // 게이트 mean band E (tokens)
    long long node_footprint_cap;     // 노드 footprint(Σlen) 캡 — count 상한보다 헤드룸(OP §4.1)

    // HBM 적합성
    double instance_a_tb;             // Instance A 점유 (KV + 가중치) 합 (TB)
    double hbm_capacity_tb;           // 가용 HBM 용량 (die-stack 높이에 비례)
    bool   hbm_fits;                  // instance_a_tb ≤ hbm_capacity_tb
};

// derive 의 튜닝 knob (CONTRACT §4 채택값 = 기본값). 문서 수치, 추정 아님.
struct DeriveOptions {
    int    decode_surplus        = 25;    // 잉여(재구성 자유도) — C 동작점(캐시 ON U-knee, OP §4.1)
    int    prefill_pool          = 60;    // depth-diversity 하한 50 + 마진 10
    int    age_cap               = 5;     // OP §3 sweep knee
    double idle_band             = 0.10;  // 진단 밴드 placeholder
    double prefill_avg_depth_frac= 0.56;  // prefill in-flight 평균 depth/ctx (OP §4.1 ~56K)
    int    node_max_surplus      = 10;    // node_max = node_min + 이 잉여 (라우팅 목표; 노드가 decode_pool 까지 로컬 top-up)
    double edge_band_tokens      = 1000;  // E = 1K (OP §7.5 채택)
    double footprint_headroom    = 1.22;  // footprint 캡 여유 (OP §4.1: 30M/24.6M = 15M/12.3M)
    // HBM die-stack 높이(dies) — 4-die SID 단위로만 쌓임(4의 배수). 실용 = 12(3 SID)·16(4 SID).
    // 용량은 JEDEC 채널밀도(die 선형)에서 산출: 16단·64스택 = 4.096 TB, 12단 = 3.072 TB.
    int    hbm_stack_height      = 16;    // 배포 16단. 12 면 용량 3/4.
};

} // namespace puls
