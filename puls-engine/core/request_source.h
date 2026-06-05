// PULS-ENGINE — 요청 출처 추상. CONTRACT.md §5.4. runtime/sim 분기점.
// 노드/글로벌 스케줄러의 admit·healing 은 이걸로만 새 요청을 얻는다.
#pragma once

namespace puls {

// 길이-도메인 요청 (토큰 내용 없음). prompt = 프롬프트 길이, dtot = 총 decode 길이.
struct PulledRequest {
    int prompt;   // < 0 이면 "없음"
    int dtot;
};

// 추상 출처. 구현:
//   - WorkloadSource (sim/validation): 분포 B + best-of-K = 무한풀 emulation.
//   - QueueSource    (runtime):        실 대기 큐에서 ideal 최근접 실 요청.
struct RequestSource {
    virtual ~RequestSource() = default;
    // ideal_tokens 근방(cap_room_tokens 적합)에서 한 요청을 당김. 없으면 PulledRequest{prompt=-1}.
    virtual PulledRequest pull_near(double ideal_tokens, long long cap_room_tokens) = 0;
};

} // namespace puls
