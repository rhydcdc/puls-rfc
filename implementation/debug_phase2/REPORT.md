# Phase-2 측정 기록 (요약)

> 동작점·물리 검증 결과를 간단히 기록. 상세 설계는 [PLAN.md](PLAN.md).

## prefill 크기별 삼중 균형 (PIM = GPU-A = B) — 512만 가능

TP=8 반영, N_dec=248, P=4 prefill 요청, KV 총량 sweep, 15% 오차 기준.

| prefill (batch당) | B 시간 | 최선 spread | 15% 균형 구간 |
|---|---|---|---|
| **512** | 101 µs | **0.6%** | **KV 21.5M~29M** (평균 ctx 87K~117K) |
| 1024 | 170 µs | 44% | **없음** |
| 2048 | 306 µs | 72% | **없음** |

**결론: prefill 512 외에는 안 됨.**

이유 — `PREFILL_ATTN = O(prefill × ctx)` 라 prefill 을 키우면 GPU-A 가 폭주:
- prefill 1024, KV 24M: PIM 98 / **GPU-A 190** / B 170 → GPU-A 단독 bottleneck, PIM 놂.
- prefill 2048, KV 24M: PIM 98 / **GPU-A 375** / B 306 → GPU-A 가 PIM 의 3.8배.

prefill 이 크면 GPU-A 가 PIM·B 를 항상 추월(PREFILL_ATTN 폭발) → KV 총량을 아무리
조절해도 셋이 안 맞음. **prefill 512 = 물리적으로 거의 유일한 균형점** (vLLM 경험적
512 와 수렴 — 둘 다 "prefill 작게 유지해야 decode/attn 과 섞임"이라는 같은 제약).

*(caveat: P=4 기준. prefill 을 더 잘게 쪼개면(P↑) 1024 가능성은 미확인 — 별도 sweep 필요.)*
