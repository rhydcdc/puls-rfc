// proto/sim — 공용 측정 (header-only). CHECKLIST §4 metrics.h.
// Σdev(평균·최악·miss분류) · p99 tail · edge%·강제율 · 캐시 히트율 · 누적절약 · 100K 유지.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "core/operating_point.h"

namespace proto {

struct Metrics {
    // 디코드 composition (배치 = decode_count_target disjoint)
    long long batches = 0, hits = 0, miss = 0;
    double    sdev_sum = 0, sdev_worst = 0, miss_dev_sum = 0;
    long long b1_cm = 0, b1_dm = 0, b2_cm = 0, b2_dm = 0;  // batch1/2 count-miss/dev-miss
    // 라우팅 / 공정성
    long long forced = 0, edged = 0;
    // 캐시
    long long returns = 0, hbm_hit = 0, ssd_reload = 0, recompute = 0;
    double    saved_rounds = 0;     // 캐시로 아낀 재로드+재계산 (라운드/시간 단위)
    // tail
    std::vector<double> ttft;       // 복귀 턴 TTFT 표본 (시간/라운드 단위)
    // 100K 유지
    double    pool_mean_sum = 0; long long pool_mean_n = 0;
    double    resid[3] = {0, 0, 0}; long long resid_n = 0;

    // 한 배치(decode_count_target 목표) 기록. b2 = 두 번째 μ-batch 여부.
    void record_batch(int count, long long sum_kv, const puls::OperatingPoint& op, bool b2) {
        ++batches;
        const double dev = std::fabs((double)sum_kv - (double)op.kv_operating_target)
                         / (double)op.kv_operating_target;
        const bool hit = (count == op.decode_count_target) && (dev <= op.idle_band);
        sdev_sum += dev;
        if (dev > sdev_worst) sdev_worst = dev;
        if (hit) {
            ++hits;
        } else {
            ++miss; miss_dev_sum += dev;
            const bool cm = (count != op.decode_count_target);
            if (!b2) { if (cm) ++b1_cm; else ++b1_dm; }
            else     { if (cm) ++b2_cm; else ++b2_dm; }
        }
    }

    // 풀 진단: 평균 live_kv + class 상주분율(short/mid/long).
    void record_pool(double mean_kv, const double cls_frac[3]) {
        pool_mean_sum += mean_kv; ++pool_mean_n;
        for (int i = 0; i < 3; ++i) resid[i] += cls_frac[i];
        ++resid_n;
    }

    double p99() const {
        if (ttft.empty()) return 0.0;
        std::vector<double> v = ttft;
        std::sort(v.begin(), v.end());
        size_t k = (size_t)std::llround(0.99 * (double)(v.size() - 1));
        return v[std::min(v.size() - 1, k)];
    }

    void print_summary(const char* label, const puls::OperatingPoint& op) const {
        (void)op;
        const double hr  = batches ? 100.0 * (double)hits / batches : 0.0;
        const double sd  = batches ? 100.0 * sdev_sum / batches : 0.0;
        const double mdd = miss ? 100.0 * miss_dev_sum / (double)miss : 0.0;
        const double chr = returns ? 100.0 * (double)hbm_hit / returns : 0.0;
        std::printf(
            "[%s] batches=%lld hit=%.2f%% Sdev=%.3f%% worst=%.2f%% missAvgDev=%.2f%% | "
            "miss b1(c/d)=%lld/%lld b2(c/d)=%lld/%lld | edge=%lld forced=%lld | "
            "returns=%lld hbmHit=%.2f%% ssdReload=%lld recompute=%lld savedR=%.0f | p99=%.1f | "
            "poolMean=%.0f resid=%.1f/%.1f/%.1f%%\n",
            label, batches, hr, sd, 100.0 * sdev_worst, mdd,
            b1_cm, b1_dm, b2_cm, b2_dm, edged, forced,
            returns, chr, ssd_reload, recompute, saved_rounds, p99(),
            pool_mean_n ? pool_mean_sum / (double)pool_mean_n : 0.0,
            resid_n ? 100.0 * resid[0] / (double)resid_n : 0.0,
            resid_n ? 100.0 * resid[1] / (double)resid_n : 0.0,
            resid_n ? 100.0 * resid[2] / (double)resid_n : 0.0);
    }
};

} // namespace proto
