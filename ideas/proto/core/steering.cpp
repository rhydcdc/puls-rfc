// PULS-ENGINE — 순수 steering + age-cap 구현. CONTRACT.md §3-② / §4.
// 근거: admission.py(steer_decode_set) + cluster_lifecycle.cpp(composeDecodeBatch/composePrefill)
//       + main_loop.py(_populate_mb_phases). canonical(§4): age-cap=가장 wait 큰 것, prefill depth +1.
#include "core/steering.h"

#include <cmath>

namespace puls {

// ── decode μ-batch steering ──────────────────────────────────────────────────
// 그리디: while n<decode_count_target ∧ S<kv_operating_target.
//  ① used 안 된 것 중 wait≥age_cap 있으면 그중 *가장 wait 큰 것* 강제 (canonical §4,
//     admission.py max(aged, key=wait) — cpp 의 first-match 아님).
//  ② 아니면 ideal=(kv_target−S)/(count_target−n) 에 |kv_length−ideal| 최소인 것 선택.
// used 마스킹 때문에 단순 선형스캔(정확성 우선). used in-place 마킹(2 μ-batch 공유 disjoint).
std::vector<int> steer_decode_set(const std::vector<Decoder>& pool,
                                  std::vector<char>& used,
                                  const OperatingPoint& op) {
    const int m = static_cast<int>(pool.size());
    std::vector<int> selected;
    long long S = 0;
    int n = 0;
    while (n < op.decode_count_target && S < op.kv_operating_target) {
        int sel = -1;
        // ① age-cap 강제: used 안 된 것 중 wait≥age_cap 인 것에서 wait 최대.
        int best_wait = -1;
        for (int i = 0; i < m; ++i) {
            if (!used[i] && pool[i].wait >= op.age_cap && pool[i].wait > best_wait) {
                best_wait = pool[i].wait;
                sel = i;
            }
        }
        // ② steering: ideal 최근접.
        if (sel < 0) {
            const double ideal =
                static_cast<double>(op.kv_operating_target - S) / (op.decode_count_target - n);
            double best_dist = 1e18;
            for (int i = 0; i < m; ++i) {
                if (used[i]) continue;
                const double dist = std::fabs(static_cast<double>(pool[i].kv_length) - ideal);
                if (dist < best_dist) {
                    best_dist = dist;
                    sel = i;
                }
            }
        }
        if (sel < 0) break;  // 후보 없음
        used[sel] = 1;
        selected.push_back(sel);
        S += pool[sel].kv_length;
        ++n;
    }
    return selected;
}

// ── prefill steering ─────────────────────────────────────────────────────────
// chunk(요청별 배정)=0 초기화. while T<budget(prefill_tokens):
//  ① wait≥age_cap ∧ chunk[i]==0 ∧ processed+chunk[i]<prompt 인 것 중 *가장 wait 큰 것* 강제
//     (spread, 요청당 1회 — main_loop.py max(aged, key=prefill_wait)).
//  ② 아니면 ideal=(prefill_kv_work_target−W)/(budget−T) 에 depth=processed+chunk[i]+1
//     (CONTRACT §4: +1) 최근접 (processed+chunk[i]<prompt 인 것만).
// chunk[sel]++; T++; W+=processed[sel]+chunk[sel] (= 방금 배정 토큰의 depth=processed+chunk_before+1).
// 반환 = chunk 벡터(pool 과 동일 길이).
std::vector<int> steer_prefill_chunks(const std::vector<PrefillReq>& pool,
                                      const OperatingPoint& op) {
    const int m = static_cast<int>(pool.size());
    std::vector<int> chunk(m, 0);
    long long W = 0;
    int T = 0;
    const int budget = op.prefill_tokens;
    while (T < budget) {
        int sel = -1;
        // ① age-cap spread: chunk==0 ∧ 잔여 프롬프트 ∧ wait≥age_cap 에서 wait 최대.
        int best_wait = -1;
        for (int i = 0; i < m; ++i) {
            if (pool[i].wait >= op.age_cap && chunk[i] == 0 &&
                pool[i].processed + chunk[i] < pool[i].prompt && pool[i].wait > best_wait) {
                best_wait = pool[i].wait;
                sel = i;
            }
        }
        // ② steering: depth(=processed+chunk+1) 최근접, 잔여 프롬프트 있는 것만.
        if (sel < 0) {
            const double ideal =
                static_cast<double>(op.prefill_kv_work_target - W) / (budget - T);
            double best_dist = 1e18;
            for (int i = 0; i < m; ++i) {
                if (pool[i].processed + chunk[i] >= pool[i].prompt) continue;
                const double depth = static_cast<double>(pool[i].processed + chunk[i] + 1);
                const double dist = std::fabs(depth - ideal);
                if (dist < best_dist) {
                    best_dist = dist;
                    sel = i;
                }
            }
        }
        if (sel < 0) break;  // 후보 없음
        ++chunk[sel];
        ++T;
        W += pool[sel].processed + chunk[sel];  // depth of 방금 배정 토큰
    }
    return chunk;
}

// ── age 갱신 ─────────────────────────────────────────────────────────────────
// selected 인덱스의 wait=0, 나머지 wait+=1.
void age_update(std::vector<Decoder>& pool, const std::vector<int>& selected) {
    std::vector<char> is_sel(pool.size(), 0);
    for (int idx : selected) is_sel[idx] = 1;
    for (std::size_t i = 0; i < pool.size(); ++i) {
        if (is_sel[i]) pool[i].wait = 0;
        else ++pool[i].wait;
    }
}

}  // namespace puls
