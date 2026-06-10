// PULS-ENGINE — 분포 B 워크로드 구현. **sim/validation 전용** (CONTRACT.md §2 · §9-2).
// 분포 B + best-of-K = 무한풀 emulation 으로 실 트래픽 부재를 대체하는 가정이다
// (ARCHITECTURE §7 honest disclosure). core 런타임 경로는 이 파일을 쓰지 않는다 —
// 실 서빙은 QueueSource 로 실 대기 큐에서 요청을 당긴다.
//
// 근거: implementation/src/cluster_routing.py (sample_b, _pull_best) +
//       implementation/analysis/cluster_balance.cpp (sampleB, pull_best).

#include "core/workload.h"

#include <cmath>
#include <limits>

namespace puls {

// 가정 분포 B (sim/validation): short 20% [1K,16K] / mid 70% [16K,256K] / long 10% [256K,1M].
// 각 구간 log-uniform = (int)exp(U×(log(hi)−log(lo))+log(lo)).
int sample_distribution_b(std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double u = uni(rng);
    double lo, hi;
    if (u < 0.20) {
        lo = 1000.0;
        hi = 16000.0;
    } else if (u < 0.90) {
        lo = 16000.0;
        hi = 256000.0;
    } else {
        lo = 256000.0;
        hi = 1000000.0;
    }
    return (int)std::exp(uni(rng) * (std::log(hi) - std::log(lo)) + std::log(lo));
}

WorkloadSource::WorkloadSource(unsigned seed, int best_of_k)
    : rng_(seed), k_(best_of_k) {}

// 무한풀 emulation: best-of-k 회 분포 B 를 뽑아 cap_room_tokens 적합한 것 중
// |length − ideal_tokens| 최소를 prompt 로 선택. 못 찾으면 prompt<0.
PulledRequest WorkloadSource::pull_near(double ideal_tokens, long long cap_room_tokens) {
    int best = -1;
    double best_dev = std::numeric_limits<double>::infinity();
    for (int i = 0; i < k_; i++) {
        int length = sample_distribution_b(rng_);
        if ((long long)length > cap_room_tokens) continue;
        double dev = std::fabs((double)length - ideal_tokens);
        if (dev < best_dev) {
            best_dev = dev;
            best = length;
        }
    }
    if (best < 0) return PulledRequest{-1, 0};
    // 길이-도메인 가정: dtot 은 무가정으로 prompt 자체로 근사(throughput 균형용 하한 1000).
    // 런타임(QueueSource)에선 실제 요청의 max_tokens 가 들어온다.
    int dtot = best > 1000 ? best : 1000;
    return PulledRequest{best, dtot};
}

} // namespace puls
