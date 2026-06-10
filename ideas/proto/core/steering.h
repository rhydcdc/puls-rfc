// PULS-ENGINE — 순수 steering + age-cap. CONTRACT.md §3-② / §4.
// 노드 스케줄러의 핵심 선택 로직. 상태 없음(풀을 받아 인덱스 선택).
#pragma once
#include <vector>
#include "operating_point.h"

namespace puls {

// 디코더(길이-도메인). kv_length = live KV(prompt+누적 decode) 추상.
struct Decoder {
    long long kv_length;
    int       wait;
};

// 프리필 요청. processed = 이미 처리한 프롬프트 토큰, prompt = 전체 프롬프트 길이.
struct PrefillReq {
    int prompt;
    int processed;
    int wait;
};

// decode μ-batch steering: ideal=(kv_target−S)/(count_target−n) closest-to-ideal 그리디.
// age-cap: wait≥age_cap 이면 강제(가장 wait 큰 것 — canonical). used[i]=true 인 것 제외.
// 반환 = 선택된 풀 인덱스. used 를 in-place 마킹(2 μ-batch 공유 disjoint).
std::vector<int> steer_decode_set(const std::vector<Decoder>& pool,
                                  std::vector<char>& used,
                                  const OperatingPoint& op);

// prefill steering: prefill_tokens 개를 depth-합(prefill_kv_work_target)으로.
// depth = processed + 배정수 + 1. age-cap spread: wait≥age_cap ∧ 이번 0배정 ∧ processed<prompt.
// 반환 = 요청별 배정 토큰 수(chunk), pool 과 같은 길이.
std::vector<int> steer_prefill_chunks(const std::vector<PrefillReq>& pool,
                                      const OperatingPoint& op);

// steering 후 wait 갱신: 선택분 wait=0, 미선택 wait+1. (호출자가 선택집합으로 적용)
void age_update(std::vector<Decoder>& pool, const std::vector<int>& selected);

} // namespace puls
