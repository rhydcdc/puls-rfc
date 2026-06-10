// PULS-ENGINE — 분포 B 워크로드. **sim/validation 전용** (CONTRACT.md §2 · §9-2).
// 실 서빙 런타임(core 경로)은 이걸 쓰지 않는다 — 실 트래픽이 들어온다.
#pragma once
#include <random>
#include "request_source.h"

namespace puls {

// 가정 분포 B: short 20% [1K,16K] / mid 70% [16K,256K] / long 10% [256K,1M], 각 구간 log-uniform.
// ARCHITECTURE §7 honest disclosure — 실 트래픽 부재 대용 가정.
int sample_distribution_b(std::mt19937& rng);

// 무한풀 emulation: 분포 B + best-of-K 로 ideal 근사. sim/validation 전용 RequestSource.
class WorkloadSource : public RequestSource {
public:
    explicit WorkloadSource(unsigned seed, int best_of_k = 200);
    PulledRequest pull_near(double ideal_tokens, long long cap_room_tokens) override;
private:
    std::mt19937 rng_;
    int k_;
};

} // namespace puls
