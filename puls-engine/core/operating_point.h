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
    int node_max;                     // 노드당 decode 풀 상한
    int node_min;                     // = 2×N_dec (2 μ-batch 바닥)
    double edge_band;                 // 게이트 mean band E (tokens)

    // HBM 적합성
    double instance_a_tb;             // Instance A KV 메모리 합 (TB)
    bool   hbm_fits;                  // ≤ substrate::PIM_CAP_TB
};

// derive 의 튜닝 knob (CONTRACT §4 채택값 = 기본값). 문서 수치, 추정 아님.
struct DeriveOptions {
    int    decode_surplus        = 10;    // 잉여(재구성 자유도) — OP §4.1
    int    prefill_pool          = 60;    // depth-diversity 하한 50 + 마진 10
    int    age_cap               = 5;     // OP §3 sweep knee
    double idle_band             = 0.10;  // 진단 밴드 placeholder
    int    staggering            = 2;     // 동시 active μ-batch
    double prefill_avg_depth_frac= 0.56;  // prefill in-flight 평균 depth/ctx (OP §4.1 ~56K)
    int    node_max_surplus      = 10;    // node_max = node_min + 이 잉여
    double edge_band_tokens      = 1000;  // E = 1K (OP §7.5 채택)
};

} // namespace puls
