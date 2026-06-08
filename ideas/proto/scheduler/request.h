// proto — 길이-도메인 요청 (멀티턴 식별 포함). 순수 데이터, 로직 없음.
// 근거: ../CHECKLIST.md, ../predictive_prepositioning_scheduler.md.
#pragma once
namespace proto {

// 길이-도메인 요청. 토큰 내용 없음. 멀티턴 캐시 추적용 id 포함.
struct Request {
    long long id = -1;       // 고유 id (멀티턴 캐시 추적). id < 0 = "없음".
    int  prompt = 0;         // 현 턴 프리필 길이 = P1(과거 캐시대상) + P2(새 메시지)
    int  max_tokens = 0;     // 이번 턴 결정론 decode 스텝 수 (= 읽을 KV 횟수). EOS 무관.
    int  cached_prefix = 0;  // P1 길이 (캐시 hit 시 프리필 면제 대상). 첫 턴 = 0.
    long long arrival = 0;   // 도착 라운드 (글로벌 age-cap = now − arrival).
    int  turn = 0;           // 턴 인덱스 (0 = 첫 턴).
};

} // namespace proto
