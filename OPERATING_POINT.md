# Phase-2 동작점 & 배치 구성 알고리즘 (확정 2026-06-01)

스케줄러가 "무엇을 배치에 담는가"의 규약. 세 자원(PIM=인스턴스 A attention / GPU-A=A
projection+prefill-attn / FFN=인스턴스 B)의 시간을 맞춰 인스턴스 간 idle 최소화.
수치는 op-time 직접 산출(PIMExecutor·compute_ffn/gpu_op_time_s, **TP=8** 반영).

> **검증 코드 (PULS 독립, composition 명중만) — puls-engine 두 sim 분담 (189 checks):**
> - [puls-engine/core/global_scheduler.cpp](puls-engine/core/global_scheduler.cpp) — **콜드스타트 분배**:
>   엣지게이트(긴 것 shed) + interleave-greedy 로 256 노드를 평균 100K 로 채우고, 남은 풀로
>   **2 disjoint 배치(on2)**가 (count, Σkv) 명중 가능함을 증명.
> - [puls-engine/sim/lifecycle.cpp](puls-engine/sim/lifecycle.cpp) — **단일 노드 생애**:
>   콜드스타트 후 steering·전이(프리필→디코드)·per-completion 힐링·age-cap 통합, 디코드(62∧6.15M)·
>   프리필(128∧12.8M) composition 유지를 증명(배포 128, 디코드 **≈99.5% 명중**·프리필 **100%**).
>
> **★ 분산이 이미 작다** — Σdev 디코드 **≈1.7%**·프리필 **≈0.1%** (실현 idle ~0).
> *이 편차 크기*가 동작점의 실질 지표 (on2 미달분도 힐링이 메움).

> **일반화 완료 (2026-06).** 본 문서의 수치(배포 prefill 128 → decode 62·6.15M·ctx≈100K,
> Instance A ≈2.77 TB 등)는 **Llama-3 70B + B200 + HBM4 16단** 기준의 *구체 예시(canonical
> instantiation)*다. 이 동작점을 만드는 *방법*(세 자원 균형 도출 + steering·cold-start·healing·
> age-cap)은 모델·GPU 무관하게 일반화되어 C++ 스케줄러
> [`puls-engine/`](puls-engine/CONTRACT.md)로 구현·검증됐으며(189 checks), 임의의 모델·GPU
> 스펙에 대해 **HBM 용량 한도 내에서 동작점을 산출**한다. 고정 = HBM4·SP-PIM·KV FP8; 변수 =
> 모델 스펙·GPU 스펙·prefill·die-stack·가중치 정밀도.

---

## 1. 고정값 (순서대로) — prefill **256** 도출 기준 (배포 동작점 = **128**, §4.1)

인과 사슬: ① prefill 토큰 수 → ② 균형 시간 X → ③ FFN batch → ④ decode 개수 → ⑤ decode KV 합.

| 순서 | 고정값 | 값 (prefill 256, 도출 기준) | 값 (prefill **128**, 배포) | 조건 자원 |
|---|---|---|---|---|
| ① | prefill 토큰/배치 | 256 | **128** | GPU-A (PREFILL_ATTN = Σ chunk×depth) |
| ② | 균형 시간 X (= 산출주기) | ~51 µs (X·L ≈ 4.1ms) | **~25.5 µs** | — |
| ③ | FFN batch | 379 토큰 | **190 토큰** | 인스턴스 B |
| ④ | **decode 개수 N_dec (제어 타깃)** | 123 (= 379 − 256) | **62** (= 190 − 128) | 인스턴스 B |
| ⑤ | **decode KV 합 (제어 타깃)** | 12.3M | **6.15M** | 인스턴스 A (PIM) |
| + | prefill KV-work (제어 타깃) | 25.6M (= 256×depth) | **12.8M** | GPU-A |
| + | 균형 ctx | ~100K (라마70B+B200 고유, §4·§4.1) | ~100K (prefill 불변·모델/GPU 종속) | — |
| + | aggregate KV (decode) | 30M → 4.92 TB — 64스택 **초과** | **13.4M decode + 프리필 in-flight = Instance A 합 2.77 TB**, **적합** §4.1 | — |

> **제어 타깃 = (개수 123, decode-KV 12.3M)** 둘 — steering 이 이 점에 수렴(§3). prefill 은
> (256 토큰, depth-합 25.6M). **배포 128 에선 절반 — (62, 6.15M)·(128, 12.8M)**; 알고리즘은
> 동일(prefill 은 *스케일 knob*, §4). steering 은 타깃을 직접 명중하므로 실현 idle ~0.
> **512 대안**: 모든 값 2배 (X 101µs, batch 759, N_dec 247, decodeKV 24.7M, prefillKVwork
> 51.2M, aggregate 60M→9.8TB). 메모리는 더 필요 — 선택 근거 §4 / §4.1.

**개수(123)는 KV 길이와 무관**(FFN 은 토큰 *개수* 만 봄). **KV 합(12.3M)은 길이의 총합**
(PIM 은 합만 봄). 둘 다 만족 ⟺ 평균 ctx ≈ 100K.

## 2. 세 자원 균형 조건

| 자원 | 시간 함수 | 조건 (prefill **128** 배포; 256 도출은 2배) |
|---|---|---|
| PIM (A attention) | f(**Σ decode KV**) | Σkv → target 6.15M |
| FFN (인스턴스 B) | f(**N_dec + prefill** = batch 토큰) | N_dec ≈ 62 (batch 190) |
| GPU-A (A proj + prefill-attn) | proj(batch) + **Σ(chunk × depth)** | prefill 128, depth합 → target 12.8M |

세 시간이 ≈25.5µs 로 모이면 overlap(F2·F3) 시 idle ≈ 0.

> **PIM 숨음.** 동작점에서 **t_pim ≤ t_gpuA**(prefill 256 측정 50.43 ≤ 52.89µs;
> 128 배포는 ~절반, 비율·margin 불변) — PIM
> decode-attn 이 GPU-A 윈도우에 *완전히 숨는다*. PREFILL_ATTN 이 t_gpuA 의 80% 라
> GPU-A 가 병목이고 PIM·FFN 이 그 뒤에 가려진다. 한때 둔 `pim_slack_safety_margin`(decode 가
> prefill compute-bound 에 못 숨을까 봐 둔 10% 헤지)은 **불필요로 판명** — 어떤 산식에도 미사용
> 이었고(타깃은 op-time 균형서 직접 도출), op-time 균형이 숨음을 확정. 기준치 재계산 불요.

## 3. former 알고리즘 — 로컬 그리디 steering + age-cap

(상수는 라마70B 배포 동작점 예시; 메커니즘은 모델 무관 — [`puls-engine/core/steering.cpp`](puls-engine/core/steering.cpp).)

**제어 타깃 = (count = 62, Σkv = 6.15M) 둘** (배포 128; 도출 기준 256 은 123·12.3M, 2배).
(avg 100K 는 이 둘의 비 = 6.15M/61.5 = 12.3M/123 으로, KV 캡 유도용 중간값일 뿐 —
*워크로드에 강제하는 값 아님*, §4.) 순수 FIFO 는 Σkv 만 잡고
*개수* 를 놓쳐 off-avg 풀에서 어긋남(검증: spread 22~30%). 그래서 매 step **"다음에 필요한
길이"** 를 계산해 그에 맞는 디코더를 고르고(steering), **너무 오래 기다린 요청은 강제 포함**
(age-cap, 공정성·FIFO 의도)한다 — 전역 통계·미래예측 없이 *로컬*:

```
한 μ-batch (decode):  # AGE_CAP = 5 (배포·클러스터 §4.1; 옛 node-scheduler sweep 은 2). 배포 128 기준
  n=0, S=0
  while n < target_count(62) and S < target_kv(6.15M) and pool:
    if (wait ≥ AGE_CAP 인 요청 있음): 가장 오래된 그것 admit   # 공정성(강제)
    else: ideal=(target_kv−S)/(target_count−n) 에 가장 가까운 디코더 admit  # steering
  나머지 대기 요청 wait += 1   # 미사용분은 다음 batch 후보
  → (62, 6.15M) 수렴. n 단조 증가라 ≤62 step 종료.
prefill 도 동일: 128 토큰을 depth-합 12.8M 되게 같은 steering+age-cap.
window=3 순차 (2 active F2/F3 overlap + 1 전이 여유).
```

- **★ 길이분산 무관 (핵심).** 거대 변종 풀(실 트래픽)에서 짧은 거+긴 거 *조합* 으로 두 타깃
  명중. 헤비/혼합/bimodal 무엇이든 동작 — avg 를 안 봄, 두 타깃만 맞춤. **age-cap 으로 강제된
  off-size(긴/짧은) 요청도 steering 이 보정**(긴 거 강제 들어오면 ideal↓ → 다음 짧은 거 다수)
  해서 배치는 여전히 (62, 6.15M). 먼저 들어온 요청은 ≤AGE_CAP+1 batch 안에 반드시 처리.
- **운영 파라미터 = target_count + target_kv + AGE_CAP 셋.** "closest-to-ideal" 자체가
  오버슈트 방지(검증: upper 가드 유무로 변종 풀 결과 불변).
- **로컬 자기보정**: 긴 걸 골랐으면 다음 `ideal`↓ → 짧은 걸. 두 축 동시 수렴. 전역 분포 안 봄.
- **검증**:
  - [`puls-engine/validation/test_steering.cpp`](puls-engine/validation/test_steering.cpp): 정규·heavy-tail·short-heavy·bimodal 전부 **N123
    Σ12.3M spread ~1%** (FIFO 는 off-avg 22~30% 실패). 원소 = 짧+중+긴 혼합(예 47+47+29).
    *(도출 256-scale 검증; 알고리즘 스케일 불변 → 배포 128 은 N62·Σ6.15M 동형, §4.1 lifecycle 실측.)*
  - [`puls-engine/validation/test_lifecycle.cpp`](puls-engine/validation/test_lifecycle.cpp): 스트리밍서 **starvation 0**(age-cap)
    — age-cap 이 모든 길이 클래스를 ≤AGE_CAP+1 batch 안에 drain → *도착한 집합 = 서빙된 집합*
    (보존). ⚠ 이건 age-cap 의 **공정성 *결과*** 이지 분포를 *타깃* 하는 게 아니다 — 배치 구성은
    여전히 avg/분포 안 보고 두 타깃만 맞춘다(길이분산 무관). steering 단독은 ideal-크기만
    cherry-pick 해 off-size 를 starve 시키므로, age-cap 이 그걸 보정해 누락 0 을 보장.
    + 매 배치 균형(spread 1.3%) + 대기 ≤3 batch.
- **AGE_CAP 트레이드오프 (sweep)**: cap↑ → steering 자유도↑ → spread↓, 단 대기(레이턴시)↑.
  cap↓ → FIFO化 → 공정/저지연이나 spread↑.  | cap1: sp3.1% | cap2: sp1.2%, 대기≤3 |
  **cap5: sp0.7%, 대기5** | cap∞: sp0.8% but **starvation(대기37)** |. → **AGE_CAP=5 채택(배포)**
  (대기 5 batch ≈ 128µs[prefill 128] ≪ TBT 8.9ms 라 레이턴시 무시 가능 → cap5 가 spread 0.7%·
  starvation 0·지연 무해의 knee. 옛 node-scheduler 는 레이턴시 보수적으로 2 였음 — 이제 5 로 통일).
- **퇴화 극단**(5분 DDoS급 롱-only 라 짧은 게 *고갈*): 짝지을 짧은 게 없어 A-bound 로 잠깐 감
  (§6.6, PIM hero 영역). 단 **age-cap 으로 안정**(starvation 0), 지나가면 복구. 실 무한-변종
  트래픽에선 미발생.

## 4. 왜 ctx 100K, 왜 prefill (배포 128)

**ctx 100K = 라마70B+B200 의 균형 ctx (경험값 아님, 이 모델·칩 고유).** prefill 에는 불변이나
모델/GPU 스펙엔 종속 — 다른 모델·칩은 puls-engine 이 재도출(§4.1). 삼중균형을 풀면 `ctx_balance = (K2+1)/K1`,
K1·K2 는 op-time 계수의 비(PIM tile rate ÷ FFN flops/tok ÷ prefill-attn flops/tok·depth ÷
proj flops/tok). **prefill 이 약분돼 사라짐** → 모든 prefill 에서 균형 ctx 가 100K (§5 스윕 B
가 실증). 이 칩(B200+HBM4+PIM)의 고유 균형 ctx.

> **★ ctx 100K 의 역할 = 타깃 *유도용*, 워크로드 *강제* 아님.** ctx 100K 로부터 제어 타깃
> (배포 128: Σkv 6.15M = 62×100K, count 62; 도출 256: 12.3M = 123×100K)을 *도출* 한다. former 는
> avg 를 보지 않고 그 두 타깃만 맞추므로, **개별 요청 길이가 어떻게 분산되든(짧/긴/혼합) 무관**(§3
> 길이분산 무관). 즉 "워크로드 arrival 평균이 100K 여야 한다"가 *아니라*, "어떤 길이분포든 KV 합
> 6.15M·개수 62 (배포) 로 조합한다".

**prefill 배포 128 (도출 기준 256, vs 512).** prefill 은 균형 ctx 가 아니라 *스케일 X* 를 정하는
knob — 작을수록 산출주기·HBM 이 절반씩 줄고 TTFT·throughput 은 불변, 단 FFN batch 가 MFU knee
위여야 한다. 한 칸씩 내릴 때(512→256→128):
- **산출주기 X**: 101 → 51 → **25.5µs** (X·L 8.1 → 4.1 → **2.0ms**).
- **HBM(aggregate)**: 60M → 30M → **15M** = 9.8 → 4.92 → **2.46TB**. **128 만 64 공식 스택(4.096TB)에
  적합**(256·512 는 초과) — §4.1. FP8 160KiB/tok.
- **TTFT 동일** — X 가 prefill 에 선형(X/prefill≈0.198 일정)이라 청크·cycle 이 상쇄,
  TTFT = prompt × 0.198 × L (prefill 무관).
- **throughput 동일** (~30k tok/s).
- prefill 작을수록 decode/prefill KV 목표 작아 **배치 구성·메모리 쉬움**.

→ **메모리는 128 이 유일 적합**(256·512 는 64 스택 초과, §4.1). **유일 risk = FFN GEMM MFU 포화**:
FFN inner dim 이 거대(K=8192, N=28672)해 wave-quant 추정상 **batch ~128 이면 포화**. 128 배포의
batch = 62 + 128 = **190 (> knee 128, 48% margin)** 이라 포화하나 256(379)보다 여유 적음. 현 모델
MFU=0.6 고정이라 knee 실측 불가(silicon 부재, ARCH "MFU plateau" deferred calibration) → **배포 128,
MFU 실측서 190 부족 판명 시 256 복귀**(256 batch 379 안전, 512 batch 759 더 안전·vLLM 수렴).

**steering 은 타깃(62, 6.15M)에 직접 명중하므로 실현 idle ~0** — former 는 `ideal` 에 가장
가까운 것을 고를 뿐, 어떤 허용 밴드로도 stop 하지 않는다.

## 4.1 HBM4 메모리 적합성 & **prefill 128 배포 동작점** (2026-06-03 라이프사이클 검증)

**노드 메모리 = Instance A 의 SP-PIM 2048 channel = HBM4 64 스택.** 공식 16-high 상한 =
32 ch × 16 Gb = **64 GB/스택** → 64 스택 = **4.096 TB**. (옛 표기 "80GB/stack·5TB" 는 스펙
초과 오기 — 정정. 80GB 면 64 스택 5.12TB 로 역산했으나 16-high 물리 상한은 64GB.)

**KV 는 FP8 저장** (160 KiB/tok = Llama-3 70B: 80층 × 8 KV head × 128 head_dim × 2(K·V)
× 1B). 가중치 FP16 은 FFN 본체가 Instance B(별도 메모리, ARCH §3.4)라 이 64 스택과 무관;
Instance A 는 QKV/O proj(~24GB)만.

| | prefill 256 | **prefill 128 (배포)** |
|---|---|---|
| decode 풀 (×100K) | 30M tok | **13.4M** (풀 134) |
| prefill in-flight (×~56K) | 8.4M tok | **3.4M** (풀 60) |
| Instance A 합 | 38.4M → **4.82 TB** | **16.8M → 2.77 TB** |
| 64 스택(4.096TB) | **초과** ✗ | **적합** ✓ (잉여 1.33 TB) |

→ **prefill 256 은 64 공식 스택에 안 들어가고**(decode 30M 만 4.92TB > 4.096TB), **prefill 128(70B)은
2.77 TB 로 들어가며 1.33 TB 남는다.** 더 큰 모델은 KV/tok 이 커져 4.096TB 를 더 소비하며, 균형 ctx 도 모델별로 재도출된다 — **'ctx
100K 불변'은 라마70B 한정 근사이고, 모델별 재도출이 일반화의 정답**(puls-engine 산출)이다.
**어느 규모(예: 405B 급) 이상은 단일 노드(64 스택)에 안 들어가며, 그 경우 노드 간 분산 서빙(TP)이
필요하다 — 다만 노드 간 통신 비용은 실측 불가라 본 RFC 의 산출 범위 밖이다.** prefill 은 *스케일
knob* 이므로 알고리즘은 그대로다.

**노드별 풀 구성 (라이프사이클 실측 확정):**

| 풀 | 크기 | 구성 | 명중 | Σdev |
|---|---|---|---|---|
| 디코드 | **134** | 124 (= 2 μ-batch × 62) + **잉여 10** | ≈99.5% | ≈1.7% |
| 프리필 | **60** | 50 (depth-diversity 하한) + **마진 10** | 100% | ≈0.1% |

> **(2026-06 sim 충실화 정정)** 이전 디코드 100%/Σdev 0.38% 는 lifecycle sim 의 힐링
> 버그(센터링 admit `ideal≈ctx_balance` → 풀 all-mid 붕괴, 긴 요청 0% → composition trivial)에서
> 나온 값이었다. canonical 힐링(per-completion `ideal=hole`, like-for-like) + 엣지 게이팅 +
> prompt-무관 현실 decode 길이 + best-of-2000 무한풀 근사로 정정하면, 분포 보존(≈20/70/10)된
> 다양 풀에서 디코드 ≈99.5%/Σdev≈1.7% (age_cap 5) — §3 sweep 의 cap5 spread(0.7%)와 정합.
> 프리필 100%/Σdev≈0.1%.

- **잉여 10 (디코드)**: hit 은 잉여 0에서도 100%; 잉여는 재구성(62+잉여→62) cherry-pick 자유도로
  *Σdev(idle 마진)* 를 조이는 knob — 늘릴수록 Σdev 가 줄다 **10 부근에서 plateau**.
  대형 모델 적재 위해 최소화하되 성능 확보 = 10.
- **마진 10 (프리필)**: 프리필 풀은 advance(~12/round)+잉여가 아니라 *0→prompt 깊이 파이프라인*.
  depth-diversity 하한 ~50(40이면 99.5%), 마진 10 = 60. 프리필은 잉여 *과대* 가 위험(age-cap
  flood, ARCH §7 참조)이지 과소가 아니라, 하한 근처가 메모리 최소이자 안전.
- **age-cap = 5**: 정의상 wait ≤ 5 batch 보장(공정성), 그 composition 비용이 ~0(위 디코드 ≈99.5% 명중)
  임을 라이프사이클서 실측. §3 sweep 도 cap5 = spread knee(0.7%) — 옛 node-scheduler 2 는
  레이턴시 보수적 선택이었고 대기 5 ≈ 128µs ≪ TBT 라 이제 **5 로 통일**.

**검증 = [puls-engine/sim/lifecycle.cpp](puls-engine/sim/lifecycle.cpp)** — 콜드스타트(KV
센터링)→프리필 steering→프리필→디코드 종속성 전이→디코드 steering→per-completion 힐링→age-cap
을 한 sim 에 통합, **종속성·age-cap 넣고도 디코드(62 ∧ 6.15M) ≈99.5%·프리필(128 ∧ 12.8M) 100% 명중**.
로직(steering·greedy·healing·age-cap·KV센터링)은 불변, 동작점 상수 6개만 절반 스케일.

> **MFU caveat (deferred calibration).** prefill 128 → FFN batch = 62 + 128 = **190 토큰**. 포화
> knee ~128 추정 위(48% 마진)라 포화하나 256(batch 379)보다 여유 적음 — silicon 부재로 미보정.
> **MFU 실측서 190 부족 판명 시 256 복귀가 fallback**(메모리는 더 필요, §4.1 표). 즉 128 = 메모리
> 최적, 256 = MFU 안전판으로 역할 분담(512↔256 프레이밍의 한 칸 아래).

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
| **128 (배포)** | **100K** | **~25.5** | **62** | **6.15M** | **12.8M** | **0.76\*** |
| 256 (도출 기준) | 100K | 51 | 123 | 12.3M | 25.6M | 0.76 |
| 512 | 100K | 101 | 247 | 24.7M | 51.2M | 0.62 |
| 1024 | 100K | 203 | 494 | 49.4M | 102.4M | 0.63 |
| 2048 | 100K | 406 | 991 | 99.1M | 204.8M | 0.39 |

> **\*** 128 의 spread 는 sweep B 재측정이 아니라 *프리필 무관*(§4: prefill 약분돼 사라짐 →
> 균형 ctx·spread 가 prefill 에 비의존)이라 256 과 동형으로 둔 값. 128 동작점의 *composition*
> 은 [puls-engine/sim/lifecycle.cpp](puls-engine/sim/lifecycle.cpp) 통합 sim(§4.1)이
> 종속성·age-cap 포함 **디코드(62∧6.15M) ≈99.5%·프리필(128∧12.8M) 100% 명중**(Σdev ≈1.7%/≈0.1%)으로 검증.

→ 균형 ctx 가 prefill 무관 100K 고정(=하드웨어 상수). prefill 은 X·배치규모만 스케일.
(옛 REPORT 의 "512만 균형, 1024+ 실패"는 decode-KV 를 25M 에 고정한 채 prefill 만 올린 측정
오류 — 각 prefill 자기 균형점에선 spread<1%. **REPORT 정정 완료** `d1e48a3` 부록 A.)

## 6. 엣지 / 구현 상태

- **짧은-평균(균일-편향) 풀**(비현실적 스트레스 케이스, 무한 변종 트래픽에선 미발생): 맞는
  길이 요청이 없어 steering 도 타깃 미달 → PIM idle = B-bound, 물리적 정상(ARCH §6.6). 고칠 대상 아님.
- **admission 구현 = former-v2 풀 모델 (완료 `54ee4d6`).** 옛 S2 `layer1`(종료 = `Σkv ≥ 목표`
  하나뿐, 개수 통제 없음)을 steering + age-cap(배포 5, §4.1) 으로 **재작성 완료** — 매 step
  `ideal=(target_kv−S)/(target_count−n)` 가장 가까운 디코더 선택(단 wait≥AGE_CAP 은 강제) →
  (개수 62, Σkv 6.15M [배포 128]; 도출 256 은 123·12.3M) 동시 수렴 + starvation 0. prefill 도
  depth-합 steering+age-cap. config: target_count·target_kv·prefill·age_cap. (= S2 가 지운
  max_batch_size 를 "FFN 개수 타깃 62"으로 의미 정정 복원.) 구성 검증은 ARCH §6.8 /
  REPORT(로컬). 통합 lifecycle 검증은 §4.1.
- prefill 값 선택은 두 제약의 균형: **메모리는 128**(64 스택 적합, §4.1) · **MFU 안전판은 256**
  (batch 379 ≫ knee). FFN MFU knee 가 미보정(silicon 부재)이라 128(batch 190)이 포화하는지는
  deferred calibration — **배포는 128, MFU 실측서 190 부족 시 256 복귀**(§4.1 caveat). 알고리즘은
  prefill 값에 비의존(family 매핑됨, 메커니즘은 어떤 prefill 이든 성립).

---

**한 줄 요약**: prefill **128(배포)** → FFN **190토큰** → **decode (개수 62 AND KV합 6.15M) 동시
타깃** (도출 기준 256 은 379·123·12.3M, 모두 2배). 스케줄러는 KV 길이를 알고 **로컬 그리디
steering**(매 step 필요한 길이에 가장 가까운 디코더 선택)으로 그 두 타깃에 수렴 — 변종 풀
spread~1%·통합 lifecycle 디코드 ≈99.5% 검증(§4.1, age-cap 5). ctx 100K 는 하드웨어 상수, prefill 은 스케일
knob(**128 배포** = HBM4 64 스택 적합 / 256 MFU 안전판).
