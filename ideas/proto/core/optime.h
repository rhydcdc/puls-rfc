// PULS-ENGINE — op-time 모델 (PIM / FFN / GPU-A). CONTRACT.md §6 수식 박제.
// 모든 모듈이 시간을 잴 때 이 함수들만 쓴다. 단위 = 마이크로초(us).
#pragma once
#include "spec.h"

namespace puls {

// PIM (Instance A attention). 입력 = Σ decode KV token 수(합). config.py pim_emulator:53-88.
double t_pim_us(long long sum_decode_kv_tokens, const HwSpec& hw);

// FFN (Instance B). 입력 = batch 토큰 개수(decode+prefill). config.py:270-274.
double t_ffn_us(int batch_total, const ModelSpec& m, const HwSpec& hw);

// GPU-A (A proj + prefill-attn). config.py:221-238,249.
//   batch_total = decode + prefill 토큰 수
//   prefill_attn_work_tokens = Σ(chunk × depth)  (depth = processed+chunk)
double t_gpu_a_us(int batch_total, long long prefill_attn_work_tokens,
                  const ModelSpec& m, const HwSpec& hw);

} // namespace puls
