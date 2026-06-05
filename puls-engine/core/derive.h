// PULS-ENGINE — ① 파라미터 산출. CONTRACT.md §3-①.
// (ModelSpec, HwSpec, prefill_tokens) → OperatingPoint. 세 자원 균형을 직접 풀어 도출.
// 문서가 손으로 한 균형(t_PIM=t_FFN=t_GPU-A)을 코드로 푼다 — 공식 발명 0.
#pragma once
#include "spec.h"
#include "operating_point.h"

namespace puls {

OperatingPoint derive_operating_point(const ModelSpec& model,
                                      const HwSpec& hw,
                                      int prefill_tokens,
                                      const DeriveOptions& opt = DeriveOptions{});

} // namespace puls
