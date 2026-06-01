# Phase-2 동작점 & 배치 구성 알고리즘 (확정 2026-06-01)

스케줄러가 "무엇을 배치에 담는가"의 규약. 세 자원(PIM=인스턴스 A attention / GPU-A=A
projection+prefill-attn / FFN=인스턴스 B)의 시간을 맞춰 인스턴스 간 idle 최소화.
수치는 op-time 직접 산출(PIMExecutor·compute_ffn/gpu_op_time_s, **TP=8** 반영).

---

## 1. 고정값 (순서대로) — prefill **256** 기본

인과 사슬: ① prefill 토큰 수 → ② 균형 시간 X → ③ FFN batch → ④ decode 개수 → ⑤ decode KV 합.

| 순서 | 고정값 | 값 (prefill 256) | 조건 자원 |
|---|---|---|---|
| ① | prefill 토큰/배치 | **256** | GPU-A (PREFILL_ATTN = Σ chunk×depth) |
| ② | 균형 시간 X | **~51 µs** (TBT ≈ X·L ≈ 4.1ms) | — |
| ③ | FFN batch | **379 토큰** | 인스턴스 B |
| ④ | **decode 개수 N_dec (제어 타깃)** | **123** (= 379 − 256) | 인스턴스 B |
| ⑤ | **decode KV 합 (제어 타깃)** | **12.3M** | 인스턴스 A (PIM) |
| + | prefill KV-work (제어 타깃) | **25.6M** (= 256×depth) | GPU-A |
| + | 균형 ctx | **~100K** (하드웨어 상수, §4) | — |
| + | aggregate KV | **~30M (2슬롯) → 5TB / 80GB·stack** | — |

> **제어 타깃 = (개수 123, decode-KV 12.3M)** 둘 — steering 이 이 점에 수렴(§3). prefill 은
> (256 토큰, depth-합 25.6M). **±10% 밴드 [11.1M,13.5M] 는 *제어값이 아니라 진단용 idle-SLA
> 경계*** (밴드 폭 ≈ 허용 최악 idle; steering 은 타깃 명중하므로 실현 idle ~0).
> **512 대안**: 모든 값 2배 (X 101µs, batch 759, N_dec 247, decodeKV 24.7M, prefillKVwork
> 51.2M, aggregate 60M→10TB). 선택 근거 §4.

**개수(123)는 KV 길이와 무관**(FFN 은 토큰 *개수* 만 봄). **KV 합(12.3M)은 길이의 총합**
(PIM 은 합만 봄). 둘 다 만족 ⟺ 평균 ctx ≈ 100K.

## 2. 세 자원 균형 조건

| 자원 | 시간 함수 | 조건 (prefill 256) |
|---|---|---|
| PIM (A attention) | f(**Σ decode KV**) | Σkv ∈ [11.1M, 13.5M] (target 12.3M) |
| FFN (인스턴스 B) | f(**N_dec + prefill** = batch 토큰) | N_dec ≈ 123 (batch 379) |
| GPU-A (A proj + prefill-attn) | proj(batch) + **Σ(chunk × depth)** | prefill 256, depth합 ∈ [23.0M, 28.2M] |

세 시간이 ≈51µs 로 모이면 overlap(F2·F3) 시 idle ≈ 0. 허용 ±10% → 가장 한가한 자원 idle ≤ ~10%.

## 3. former 알고리즘 — 로컬 그리디 steering + age-cap

**제어 타깃 = (count = 123, Σkv = 12.3M) 둘.** (avg 100K 는 이 둘의 비 = 12.3M/123 으로,
KV 캡 유도용 중간값일 뿐 — *워크로드에 강제하는 값 아님*, §4.) 순수 FIFO 는 Σkv 만 잡고
*개수* 를 놓쳐 off-avg 풀에서 어긋남(검증: spread 22~30%). 그래서 매 step **"다음에 필요한
길이"** 를 계산해 그에 맞는 디코더를 고르고(steering), **너무 오래 기다린 요청은 강제 포함**
(age-cap, 공정성·FIFO 의도)한다 — 전역 통계·미래예측 없이 *로컬*:

```
한 μ-batch (decode):  # AGE_CAP = 2 (기본)
  n=0, S=0
  while n < target_count(123) and S < target_kv(12.3M) and pool:
    if (wait ≥ AGE_CAP 인 요청 있음): 가장 오래된 그것 admit   # 공정성(강제)
    else: ideal=(target_kv−S)/(target_count−n) 에 가장 가까운 디코더 admit  # steering
  나머지 대기 요청 wait += 1   # 미사용분은 다음 batch 후보
  → (123, 12.3M) 수렴. n 단조 증가라 ≤123 step 종료.
prefill 도 동일: 256 토큰을 depth-합 25.6M 되게 같은 steering+age-cap.
window=3 순차 (2 active F2/F3 overlap + 1 전이 여유).
```

- **★ 길이분산 무관 (핵심).** 거대 변종 풀(실 트래픽)에서 짧은 거+긴 거 *조합* 으로 두 타깃
  명중. 헤비/혼합/bimodal 무엇이든 동작 — avg 를 안 봄, 두 타깃만 맞춤. **age-cap 으로 강제된
  off-size(긴/짧은) 요청도 steering 이 보정**(긴 거 강제 들어오면 ideal↓ → 다음 짧은 거 다수)
  해서 배치는 여전히 (123, 12.3M). 먼저 들어온 요청은 ≤AGE_CAP+1 batch 안에 반드시 처리.
- **운영 파라미터 = target_count + target_kv + AGE_CAP 셋.** 상한·하한 밴드는 제어 아님 —
  "closest-to-ideal" 자체가 오버슈트 방지(검증: upper 가드 유무로 변종 풀 결과 불변).
- **로컬 자기보정**: 긴 걸 골랐으면 다음 `ideal`↓ → 짧은 걸. 두 축 동시 수렴. 전역 분포 안 봄.
- **[11.1M,13.5M] 밴드 = 진단용 idle-SLA 라벨**(±10%→idle≤10%), 제어값 아님.
- **검증**:
  - [proto_steering.py](proto_steering.py): 정규·heavy-tail·short-heavy·bimodal 전부 **N123
    Σ12.3M spread ~1%** (FIFO 는 off-avg 22~30% 실패). 원소 = 짧+중+긴 혼합(예 47+47+29).
  - [proto_steering_fair.py](proto_steering_fair.py): 스트리밍서 **서빙 분포 = arrival 분포**
    (starvation 0, 모든 클래스 drain) + 매 배치 균형(spread 1.3%) + 대기 ≤3 batch.
- **AGE_CAP 트레이드오프 (sweep)**: cap↑ → steering 자유도↑ → spread↓, 단 대기(레이턴시)↑.
  cap↓ → FIFO化 → 공정/저지연이나 spread↑.  | cap1: sp3.1% | **cap2: sp1.2%, 대기≤3** |
  cap5: sp0.7%,대기5 | cap∞: sp0.8% but **starvation(대기37)** |. → **AGE_CAP=2 채택**
  (오래 기다리면 spread 떨어지지만 레이턴시 길어져 안 됨 → 2 가 균형·공정·지연 sweet spot).
- **퇴화 극단**(5분 DDoS급 롱-only 라 짧은 게 *고갈*): 짝지을 짧은 게 없어 A-bound 로 잠깐 감
  (§6.6, PIM hero 영역). 단 **age-cap 으로 안정**(starvation 0), 지나가면 복구. 실 무한-변종
  트래픽에선 미발생.

## 4. 왜 ctx 100K, 왜 prefill 256

**ctx 100K = 하드웨어 상수 (경험값 아님).** 삼중균형을 풀면 `ctx_balance = (K2+1)/K1`,
K1·K2 는 op-time 계수의 비(PIM tile rate ÷ FFN flops/tok ÷ prefill-attn flops/tok·depth ÷
proj flops/tok). **prefill 이 약분돼 사라짐** → 모든 prefill 에서 균형 ctx 가 100K (§5 스윕 B
가 실증). 이 칩(B200+HBM4+PIM)의 고유 균형 ctx.

> **★ ctx 100K 의 역할 = 타깃 *유도용*, 워크로드 *강제* 아님.** ctx 100K 로부터 제어 타깃
> (Σkv 12.3M = 123 × 100K, count 123)을 *도출* 한다. former 는 avg 를 보지 않고 그 두 타깃만
> 맞추므로, **개별 요청 길이가 어떻게 분산되든(짧/긴/혼합) 무관**(§3 길이분산 무관). 즉 "워크로드
> arrival 평균이 100K 여야 한다"가 *아니라*, "어떤 길이분포든 KV 합 12.3M·개수 123 으로 조합한다".

**prefill 256 기본 (vs 512).** prefill 은 균형 ctx 가 아니라 *스케일 X* 를 정하는 knob. 256 이:
- **TBT 절반** (X 51 vs 101µs → 4.1 vs 8.1ms).
- **HBM 절반** (aggregate ~30M→5TB vs 60M→10TB).
- **TTFT 동일** — X 가 prefill 에 선형(X/prefill≈0.198 일정)이라 청크 2배·cycle 절반이
  상쇄, TTFT = prompt × 0.198 × L (prefill 무관).
- **throughput 동일** (~30k tok/s).
- decode/prefill KV 목표가 작아 **배치 구성도 쉬움**(변동·메모리 적음).

→ 256 이 512 대비 TBT·HBM 을 반으로 줄이면서 TTFT·throughput 손해 0. **유일 risk = FFN GEMM
MFU 포화**: batch 379 가 텐서코어를 다 채우나? FFN inner dim 이 거대(K=8192, N=28672)해
wave-quant 추정상 batch ~128 이면 포화(379 는 충분) → **256 채택**. 단 현 모델은 MFU=0.6
고정이라 knee 를 못 봄 = **실측 불가(silicon 부재, ARCH "MFU plateau" deferred calibration).**
**그래서 512 는 FFN 포화 불가 판명 시 대안**(batch 759 확실 포화, vLLM 수렴).

**오차 밴드 = ±10% (진단용 idle SLA 경계, 제어값 아님).** 밴드 폭 ≈ 허용 최악 idle: ±10%→
edge idle ~8.6~10.6%, ±15%→~12.4~15.4%. **steering 은 타깃(123, 12.3M)에 명중하므로 실현
idle ~0** — 밴드는 "이 안이면 idle≤10%" 라는 진단 라벨일 뿐(former 가 밴드로 stop 하지 않음).
밴드 폭은 ARCH §6.4 deadband=2σ_total 근거이나 σ 실측 불가라 deferred
calibration; 10% 는 그 placeholder. (15%/20% 도 동작 가능 — calibration 때 조정.)

## 5. 스윕 결과 (도출 근거 — 보존)

### 5.1 스윕 A — ctx 스윕 (prefill 512): 100K 가 유일 삼중-균형점

각 ctx 에서 "FFN=GPU-A 되는 N_dec"를 찾고 그때 t_PIM 비교한 spread:

| ctx | N_dec* | X(=FFN=GPU-A) | t_PIM | Σkv | spread% |
|---|---|---|---|---|---|
| 40K | 0 | 57µs | 0 | 0 | 100 |
| 80K | 96 | 81µs | 31.5 | 7.7M | 61 |
| **100K** | **247** | **101µs** | **100.7** | **24.7M** | **0.6** |
| 120K | 399 | 122µs | 195 | 47.9M | 38 |
| 200K | 1005 | 202µs | 819 | 201M | 75 |

→ spread 가 ctx=100K 에서만 ≈0. 다른 ctx 는 FFN=GPU-A 맞춰도 PIM 이 크게 어긋남.

### 5.2 스윕 B — prefill × ctx: 모든 prefill 이 ctx 100K 에서 균형

| prefill | 균형 ctx | X(µs) | N_dec | decode-KV | prefill-KV-work | spread% |
|---|---|---|---|---|---|---|
| **256** | **100K** | **51** | **123** | **12.3M** | **25.6M** | **0.76** |
| 512 | 100K | 101 | 247 | 24.7M | 51.2M | 0.62 |
| 1024 | 100K | 203 | 494 | 49.4M | 102.4M | 0.63 |
| 2048 | 100K | 406 | 991 | 99.1M | 204.8M | 0.39 |

→ 균형 ctx 가 prefill 무관 100K 고정(=하드웨어 상수). prefill 은 X·배치규모만 스케일.
(REPORT 의 "512만 균형, 1024+ 실패"는 decode-KV 를 25M 에 고정한 채 prefill 만 올린 측정
오류 — 각 prefill 자기 균형점에선 spread<1%. REPORT 정정 대상.)

## 6. 엣지 / 구현 상태

- **짧은-평균(균일-편향) 풀**(비현실적 스트레스 케이스, 무한 변종 트래픽에선 미발생): 맞는
  길이 요청이 없어 steering 도 타깃 미달 → PIM idle = B-bound, 물리적 정상(ARCH §6.6). 고칠 대상 아님.
- **현 `admission.layer1`(S2)**: 종료 = `Σkv ≥ 목표` 하나뿐(개수 통제 없음). **former-v2 에서
  steering + age-cap(2) 으로 재작성** — 매 step `ideal=(target_kv−S)/(target_count−n)` 가장 가까운
  디코더 선택(단 wait≥AGE_CAP 은 강제) → (개수 123, Σkv 12.3M) 동시 수렴 + starvation 0. 풀
  길이-인덱싱(버킷) + wait 추적. prefill 도 depth-합 steering+age-cap. config: target_count·
  target_kv·prefill·age_cap. (= S2 가 지운 max_batch_size 를 "FFN 개수 타깃 123"으로 의미 정정 복원.)
- prefill 최적값(256 vs 512)은 FFN MFU knee 에 의존 → deferred calibration. 알고리즘은
  prefill 값에 비의존(family 매핑됨, 메커니즘은 어떤 prefill 이든 성립).

---

**한 줄 요약**: prefill 256 → FFN 379토큰 → **decode (개수 123 AND KV합 12.3M) 동시 타깃**.
스케줄러는 KV 길이를 알고 **로컬 그리디 steering**(매 step 필요한 길이에 가장 가까운 디코더
선택)으로 그 두 타깃에 수렴 — 변종 풀에서 spread~1% 검증. ctx 100K 는 하드웨어 상수,
prefill 은 스케일 knob(256 기본 / 512 포화 대안). 밴드는 진단용 idle-SLA(제어값 아님).
