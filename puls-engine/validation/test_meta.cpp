// 검증 — 구조·계약 회귀 (메타). CONTRACT.md §7 메타/회귀 · §9 불변식.
// 동작점 리터럴 금지(§9-5)는 코드 grep 불가 → *행동* 으로 증명한다:
//   ① derive 출력이 입력 변수에 반응 — 모델 hidden 바꾸면 ctx_balance/N_dec 달라짐(하드코딩 아님).
//   ② 고정/변수 경계 — substrate 상수가 컴파일 상수로 존재, ModelSpec/HwSpec 에 그 값이 안 섞임
//      (필드 부재는 컴파일되는 것으로 충분 + 몇 개 static 확인).
//   ③ canonical: per-completion(toxic-fit) — 긴 요청 보존 1 케이스. age-cap=wait-max 1 케이스.
//   ④ prefill 스케일 불변 — derive(.,.,128) vs (.,.,256) → N_dec≈2배, ctx 불변.
#include "test_framework.h"
#include "core/spec.h"
#include "core/derive.h"
#include "core/steering.h"
#include <cstdio>
#include <vector>
#include <type_traits>

using namespace puls;

int main() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};

    // ── ① derive 가 입력 변수에 반응(하드코딩 아님) ─────────────────────────────
    // 모델 hidden 을 바꾸면 op-time 계수가 달라져 ctx_balance/N_dec 가 재도출된다.
    OperatingPoint base = derive_operating_point(llama, b200, 128);
    ModelSpec wide = llama; wide.hidden = llama.hidden * 2;      // hidden 2배
    OperatingPoint op_wide = derive_operating_point(wide, b200, 128);
    std::printf("react-hidden: base ctx=%.0f N=%d | wide ctx=%.0f N=%d\n",
                base.ctx_balance, base.decode_count_target,
                op_wide.ctx_balance, op_wide.decode_count_target);
    // 적어도 하나(ctx_balance 또는 N_dec)는 입력 변화에 반응해야 한다 = 리터럴 박제 아님.
    CHECK(op_wide.ctx_balance != base.ctx_balance ||
          op_wide.decode_count_target != base.decode_count_target,
          "derive reacts to model.hidden (not a hardcoded literal)");

    // ffn_intermediate 도 마찬가지로 N_dec/ctx 에 반응(FFN op-time 항).
    ModelSpec fat_ffn = llama; fat_ffn.ffn_intermediate = llama.ffn_intermediate * 2;
    OperatingPoint op_ffn = derive_operating_point(fat_ffn, b200, 128);
    CHECK(op_ffn.ctx_balance != base.ctx_balance ||
          op_ffn.decode_count_target != base.decode_count_target,
          "derive reacts to model.ffn_intermediate (not hardcoded)");

    // HW(mfu) 도 반응.
    HwSpec slow = b200; slow.gpu_mfu = b200.gpu_mfu * 0.5;
    OperatingPoint op_slow = derive_operating_point(llama, slow, 128);
    CHECK(op_slow.ctx_balance != base.ctx_balance ||
          op_slow.decode_count_target != base.decode_count_target ||
          op_slow.balance_time_us != base.balance_time_us,
          "derive reacts to hw.gpu_mfu (not hardcoded)");

    // ── ② 고정/변수 경계: substrate 상수는 컴파일 상수, spec 구조체에 안 섞임 ──────
    // substrate 상수가 컴파일 상수로 실재(몇 개 static 확인). 값까지 박제(CONTRACT §1).
    static_assert(substrate::PIM_TILE_TIME_FP8_NS == 267.0, "PIM tile time FP8 = 267 ns");
    static_assert(substrate::PIM_TILE_ROWS == 32, "PIM tile rows = 32");
    static_assert(substrate::PIM_CAP_TB == 4.40, "PIM cap = 4.40 TB");
    static_assert(substrate::KV_BYTES_PER_ELEM == 1, "FP8 KV = 1 byte/elem (고정)");
    // ModelSpec/HwSpec 의 크기 = 선언된 변수 필드만 — 고정 substrate 값이 필드로 안 샘.
    CHECK(sizeof(ModelSpec) == sizeof(int) * 7,
          "ModelSpec = 6 dims + weight precision (변수만; substrate 비혼입)");
    static_assert(std::is_trivially_copyable<ModelSpec>::value, "ModelSpec stays a plain spec");
    // HwSpec = 2 double + num_gpus_a/b (변수 — 배포 8, 다른 구성 가능). 정렬 패딩 하한·상한 핀.
    CHECK(sizeof(HwSpec) >= sizeof(double) * 2 + sizeof(int) * 2 &&
          sizeof(HwSpec) <= sizeof(double) * 3,
          "HwSpec = tflops+mfu+num_gpus_a/b (variables; gpus default 8)");
    // kv_bytes_per_token 은 변수 필드 × substrate 상수로 계산 — substrate 값이 spec 에 박히지 않음.
    long long expect_kv = (long long)2 * llama.num_kv_heads * llama.head_dim * 1 * llama.num_layers;
    CHECK(llama.kv_bytes_per_token() == expect_kv,
          "kv_bytes/token derives from variable fields x substrate constants");
    // 가중치 정밀도는 *동적* — FP8/FP16/FP32 로 instance_a 가중치 바이트가 비례.
    ModelSpec w_fp8 = llama;  w_fp8.weight_bytes_per_elem  = 1;
    ModelSpec w_fp32 = llama; w_fp32.weight_bytes_per_elem = 4;
    std::printf("weight-precision: fp8=%.1fGB fp16=%.1fGB fp32=%.1fGB\n",
                w_fp8.instance_a_weight_bytes() / 1e9, llama.instance_a_weight_bytes() / 1e9,
                w_fp32.instance_a_weight_bytes() / 1e9);
    CHECK(w_fp8.instance_a_weight_bytes() * 2 == llama.instance_a_weight_bytes() &&
          w_fp32.instance_a_weight_bytes() == llama.instance_a_weight_bytes() * 2,
          "weight precision is dynamic (FP8/FP16/FP32 scales weight bytes)");

    // ── ③ canonical: per-completion 힐링(toxic-fit) — 긴 요청 보존 ───────────────
    // steering 은 batched 평균을 안 쓴다(구조). 행동검증: ideal=가장 작은 잔여라도 긴 요청이
    // 굶지 않고 명중 집합에 포함 — kv_target 채우려면 큰 요소가 필수라 toxic-fit 으로 보존된다.
    {
        OperatingPoint op = base;
        // 풀: 다수의 짧은 + 소수의 긴(=ctx 보다 큰) 요청. batched(평균) 힐링이면 긴 게 굶지만,
        // closest-to-ideal(per-completion 의미)은 kv 합을 맞추려 큰 것을 끌어들인다.
        std::vector<Decoder> pool;
        const long long longkv = (long long)(op.ctx_balance * 3); // 평균보다 훨씬 긴 요청
        int n_long = 0;
        for (int i = 0; i < op.decode_count_target * 2; ++i) {
            if (i % 4 == 0) { pool.push_back(Decoder{longkv, 0}); ++n_long; }
            else            { pool.push_back(Decoder{(long long)(op.ctx_balance / 2), 0}); }
        }
        std::vector<char> used(pool.size(), 0);
        std::vector<int> sel = steer_decode_set(pool, used, op);
        int long_in_sel = 0;
        for (int idx : sel) if (pool[idx].kv_length == longkv) ++long_in_sel;
        std::printf("toxic-fit: pool_long=%d selected=%zu long_in_sel=%d\n",
                    n_long, sel.size(), long_in_sel);
        // toxic-fit: 긴 요청이 선택집합에 보존(굶지 않음). batched 평균이면 0 이었을 것.
        CHECK(long_in_sel > 0, "toxic-fit: long requests preserved in steering set (not starved)");
    }

    // ── ③ canonical: age-cap = 가장 wait 큰 요청 (first-match 아님) ──────────────
    {
        OperatingPoint op = base;
        // 두 요청이 wait≥age_cap. canonical 은 *가장 wait 큰* 것을 먼저 강제해야 한다.
        // 인덱스 순서상 first-match(작은 wait)가 아니라 max-wait 가 먼저 뽑히는지로 구별.
        std::vector<Decoder> pool;
        // idx0: aged but smaller wait(=age_cap). idx1: aged with larger wait(=age_cap+3).
        pool.push_back(Decoder{(long long)op.ctx_balance, op.age_cap});      // first-match 후보
        pool.push_back(Decoder{(long long)op.ctx_balance, op.age_cap + 3});  // wait-max 후보
        // 나머지는 wait 작은 정상 요청으로 채움(steering 이 가져가도 무방).
        for (int i = 0; i < op.decode_count_target; ++i)
            pool.push_back(Decoder{(long long)(op.ctx_balance), 0});
        std::vector<char> used(pool.size(), 0);
        std::vector<int> sel = steer_decode_set(pool, used, op);
        // 첫 선택이 wait-max(idx1)여야 한다 — first-match(idx0) 면 canonical 위반.
        CHECK(!sel.empty(), "age-cap: at least one selection");
        std::printf("age-cap: first_pick_idx=%d (expect 1=wait-max, not 0=first-match)\n",
                    sel.empty() ? -1 : sel[0]);
        CHECK(!sel.empty() && sel[0] == 1, "age-cap forces the MAX-wait request (canonical, not first-match)");
    }

    // ── ④ prefill 스케일 불변: P=128 vs 256 → N_dec≈2배, ctx 불변 ────────────────
    OperatingPoint p128 = derive_operating_point(llama, b200, 128);
    OperatingPoint p256 = derive_operating_point(llama, b200, 256);
    std::printf("prefill-scale: P128 ctx=%.0f N=%d | P256 ctx=%.0f N=%d\n",
                p128.ctx_balance, p128.decode_count_target,
                p256.ctx_balance, p256.decode_count_target);
    CHECK_REL(p256.ctx_balance, p128.ctx_balance, 0.05, "ctx_balance prefill-invariant");
    CHECK_NEAR(p256.decode_count_target, 2 * p128.decode_count_target, 6,
               "N_dec scales ~2x with prefill (128->256)");
    CHECK_REL(p256.kv_operating_target, 2.0 * (double)p128.kv_operating_target, 0.10,
              "kv_target scales ~2x with prefill");

    // ── ⑤ footprint 캡 헤드룸 (회귀 핀) ─────────────────────────────────────────
    // cold-start/heal 의 footprint 캡은 count 상한(node_max×ctx)보다 헤드룸이 있어야
    // 노드가 긴 꼬리를 담은 채 node_max 까지 차서 on2 가 유지된다. node_max×ctx 로 조이면
    // on2 가 88%로 회귀(측정). 문서값 15M@128 / 30M@256 = 2×kv_target×1.22(OP §4.1).
    {
        const long long tight = (long long)base.node_max * (long long)base.ctx_balance;
        std::printf("footprint-cap: cap=%lld tight(node_max*ctx)=%lld ratio=%.3f\n",
                    base.node_footprint_cap, tight,
                    (double)base.node_footprint_cap / (double)tight);
        CHECK(base.node_footprint_cap > tight,
              "footprint cap has headroom over node_max*ctx (else on2 regresses ~88%)");
        CHECK_REL(base.node_footprint_cap, 2.0 * (double)base.kv_operating_target * 1.22, 0.02,
                  "footprint cap ~= 2x kv_target x headroom (OP 4.1: 15M@128 / 30M@256)");
    }

    return puls_test::summary();
}
