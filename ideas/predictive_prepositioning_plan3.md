# Plan 3 — 충실한 TBT/TTFT: A→B 의존성 · HBM 컨텐션 + 비용모델 교정

> Plan 2 회고에서 드러난 세 갭을 메워 **진짜 TBT/TTFT/SLO goodput** 을 얻는다.
> ① reload BW 가 HBM급이라 캐시 가치가 안 드러남 ② 콜드스타트 시딩 불일치로 C 캐시 비교 오염
> ③ TBT = max(pim,gpua,ffn) 은 무컨텐션·완전오버랩 상한 — A→B→A 의존성·HBM 컨텐션 미반영.

---

## 1. 교정 1 — reload BW 제값 (캐시가 의미를 갖게)
- 현재 `offload_bw = 5e9 B/round ≈ 2.46 TB/s` = **HBM급**(오기). SSD 는 7~14 GB/s.
- → `offload_bw ≈ 2e7 B/round`(SSD ~10 GB/s) 로 교정. *가정 라벨, 스윕 가능.* (CPU-DRAM 택1 시 ~1.3e8.)
- 효과: miss(reload ∝ 길이 / recompute) 비용이 실값(100K KV reload ≈ 1.7s) → **hit(B)의 우위가 TTFT·goodput 에 비로소 드러남.** (현재는 reload 6.6ms 라 큐 대기에 묻힘.)

## 2. 교정 2 — 콜드스타트 시딩 일치 (공정 캐시 비교)
- 현 버그: A/C 는 콜드스타트를 *자체 draw*(sample_distribution_b)로 채워 **시드한 큐(~8800)를 안 빼냄** → 큐 만성 적체 → 복귀 대기 ≫ evict_age → C 히트 붕괴(3%). B 는 큐에서 콜드스타트해 큐를 ~300 으로 배수.
- → **A/C 도 큐에서 콜드스타트**(또는 시드량 일치)로 초기 적체 제거.
- 가설: 수정 후 **B ≈ C 히트(용량 차 ~7%만)** — C 의 낮은 히트가 버그였음 확인. (잔차 배수차 있으면 측정.)

## 3. 핵심 — A→B→A 의존성 + HBM 컨텐션 (`t_pim ≤ t_gpu_a` 조건)
세 자원을 독립 max() 로 보던 걸, **인스턴스 단위 의존성**으로 바꾼다:
- **인스턴스 A 지연** = `max(t_pim, t_gpu_a)` (PIM ∥ GPU-A 오버랩) → 끝나야 **인스턴스 B(FFN) 시작** (A→B 의존). 다음 층은 B→A.
- **컨텐션-무료 조건 = `t_pim ≤ t_gpu_a`** (PIM 이 GPU-A 그림자에 숨어 HBM BW 충돌 흡수; 더 작을수록 안전 = PIMwin2 지표 그 자체). 이때 A_time = t_gpu_a.
- **위반 (`t_pim > t_gpu_a`, PIM 노출) → 컨텐션 페널티**: PIM 이 길게 도는 동안 GPU-A 가 *다음 배치 QKV 백필*하며 HBM 공유 → 노출분이 느려짐.
  - 단순형: `A_time = max(t_pim, t_gpu_a) + β · max(0, t_pim − t_gpu_a)`, **β ∈ [0,1] 스윕** (0 = 무컨텐션 = 현 모델, 1 = 노출분 완전 직렬화). FFN 은 별도 메모리(ARCH §3.4)라 컨텐션 무관 → 조건은 PIM↔GPU-A 만.
- **TBT(처리율, 더블버퍼링)** = `max(A_time, t_ffn) × num_layers`.
  - 의존성*만*으론 throughput = max(pim,gpua,ffn) 그대로 → **차이를 만드는 건 β 페널티.** (B 노출 99.6% → 큰 페널티, A/C ~절반.)
- **컨텐션 노출 카운트**: `t_pim > t_gpu_a` 비율(= 1 − PIMwin2) + 누적 노출 페널티 시간.
- 주의: 동작점(prefill 128)서 `t_pim ≈ t_gpu_a` 라 A/C 도 ~절반 노출, B 거의 항상 — **모델이 "동작점이 PIM-hiding 가장자리"임을 드러냄.**

## 4. 재측정 / 스윕
- 교정·컨텐션 반영 후 A/B/C 의 **TBT(평균·p99) · TTFT(평균·p99) · SLO goodput · PIM-노출율** 재측정.
- 스윕: 글로벌 age-cap · 노드 age-cap · evict_age · **offload_bw** · **컨텐션 β**.
- 답할 질문:
  1. reload 제값 후 **캐시(B)가 goodput 에 얼마나 기여**하나.
  2. 시딩 수정 후 **B vs C 히트 차이 = 용량뿐인가**.
  3. 컨텐션(β) 반영 시 **B 의 PIM-노출이 TBT/goodput 을 얼마나 깎나** — **idle(Σdev) → PIM노출 → 컨텐션 → TBT → goodput** 사슬 완성.

## 5. 경계
- `offload_bw · β · SLO 임계 · think_gap · gone_age` 는 가정 라벨(스윕 또는 고정). TBT/TTFT 절대값은 B200 + Llama70B optime 종속(HW 바뀌면 재도출).
- idle/Σdev 는 *목표 아님* — PIM-노출·컨텐션·TBT 경유로만 goodput 에 영향. 최종 판단 = SLO goodput.
