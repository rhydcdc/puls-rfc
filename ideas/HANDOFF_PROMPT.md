# 핸드오프 프롬프트 — 현재 스케줄러 정확히 파악 + Pre-Positioning 아이디어 프로토타입

> 새 세션의 Claude 에게: 아래를 **그대로 임무로** 받아라. 이 프로젝트(`c:\Users\rhs02\Desktop\puls-rfc`)는
> GPU + PIM 기반 LLM 서빙 스케줄러 RFC("PULS")다. 너의 일은 **(1) 현재 스케줄러 로직을 오해 없이
> 완전히 파악**하고 **(2) `ideas/predictive_prepositioning_scheduler.md` 의 새 설계를 `ideas/` 안에
> 프로토타입으로 구현·측정**하는 것이다.

---

## 0. 절대 원칙 (먼저 읽어라)

1. **추측·단정 금지. 빌드·실행으로 측정해 검증하라.** 이 사용자는 hand-waving 과 틀린 확신을 즉시 잡아낸다. "아마 ~일 것이다"로 결론 내지 말고, sim 을 빌드·실행해 SUMMARY 숫자로 말하라. 가설은 가설이라 명시하고 스윕으로 검증하라.
2. **`puls-engine/core/` 는 정식(canonical) 로직 — 절대 건들지 마라.** `puls-engine/sim/` 은 검증용 sim. 새 프로토타입은 **`ideas/` 안에서만** 만들어라(`ideas/` 는 gitignore — 로컬 전용).
3. 단위 일은 한국어로 답하되, 코드 주석은 주변 스타일에 맞춰라.

---

## 1. 현재 스케줄러 — 반드시 정독할 canonical 소스 (순서대로)

| 파일 | 내용 |
|---|---|
| `puls-engine/CONTRACT.md` | **단일 진실원.** canonical 결정 표(§4): 힐링=ideal=hole, age-cap=max-wait, prefill depth=processed+chunk+1, batched/centering 힐링 **금지**. |
| `puls-engine/core/steering.cpp` `.h` | `steer_decode_set`(디코드: `ideal=(kv_target−S)/(count_target−n)` 최근접 그리디, age-cap=wait 최대 강제, 2 μ-batch `used` 공유 disjoint), `steer_prefill_chunks`(프리필: depth=processed+chunk+1, age-cap spread). |
| `puls-engine/core/node_scheduler.cpp` `.h` | `advance_round`(2 μ-batch 구성 → 선택분 dec++/wait=0·미선택 wait++ → dec≥dtot retire → **per-completion `ideal=hole` 힐링**), `admit_centered`(콜드/센터링 primitive — 힐링 아님). |
| `puls-engine/core/global_scheduler.cpp` `.h` | `gate`(**엣지 게이팅**: 최장 요청을 shed 해 kept 평균 ≤ ctx_balance+edge_band), `cold_start`(interleave-greedy `min|추가후 mean−ctx_balance|`, can_fit 게이트), `heal_node`(ideal=hole), `onpoint_batches`(검증용 disjoint-K 명중 체크). |
| `puls-engine/core/workload.cpp` `.h` | `sample_distribution_b`(분포 B: short 20%[1K,16K]/mid 70%[16K,256K]/long 10%[256K,1M], 평균 ~116K), `WorkloadSource`(best-of-k 무한풀 emulation — **sim 전용**, 실서빙은 `QueueSource`). |
| `puls-engine/core/operating_point.h`, `derive.*` | `OperatingPoint` 필드, `derive_operating_point(model, hw, prefill)`. |
| `OPERATING_POINT.md` | 배포 동작점: prefill 128 → **count 62 ∧ kv 6.15M**, prefill-work 12.8M, **ctx_balance ~100K**, **decode_pool 134(=124+잉여10)**, prefill_pool 60(=50+마진10), **age_cap 5**, idle_band 0.10, edge_band, **HBM 4.096TB 중 2.77 사용·1.33TB 잉여**. §3 알고리즘 / §4 도출 / §4.1 풀·메모리. |
| `ARCHITECTURE.md` `.ko.md` | 3-자원(Instance A = PIM decode-attn + GPU-A proj/prefill-attn, Instance B = FFN), F2(proj‖attn)·F3(A‖B inter-instance 파이프라인), **2-active μ-batch staggering**, 80 layers(Llama70B), §7 클러스터 엣지-게이팅 라우팅. |
| `puls-engine/sim/lifecycle.cpp` | 통합 검증 sim(콜드스타트 gate+cold_start → 프리필 steering → prefill→decode 전이 → 디코드 steering 2 μ-batch → advance → retire → ideal=hole 힐링 → 계측). **이번 세션서 충실화 완료.** |

빌드: `g++ -std=c++17 -O2 -Wall -Wextra -I. -Ivalidation sim/lifecycle.cpp core/optime.cpp core/derive.cpp core/steering.cpp core/node_scheduler.cpp core/global_scheduler.cpp core/workload.cpp -o build/puls_lifecycle.exe` (cwd = `puls-engine/`). 실행: `./build/puls_lifecycle.exe <ITERS> <Z> <age_cap> <best_of_k>` → `[SUMMARY]` 한 줄에 hit/Σdev/edge/aged/forced/분포 다 나옴.

---

## 2. ⚠️ 자주 틀리는 지점 (지난 세션서 실제로 다 틀렸다 — 반복 금지)

1. **힐링 = per-completion `ideal=hole`(like-for-like / toxic-fit), 센터링 아님.** 센터링 힐링(`ideal≈ctx_balance`)은 CONTRACT §4 **금지** — 풀을 all-mid 로 붕괴(긴 요청 0%)시켜 composition 을 trivial 하게 만든다. 문서의 옛 "디코드 100%/Σdev 0.38%"가 바로 이 버그 산물이었고, 이번에 정정해 **충실값 ≈99.5%/1.7%** 가 됐다.
2. **decode_pool 134 = 124(2 μ-batch×62) + 잉여 10 = 동작점이다. 임의로 키우지 마라.** 노드 풀을 키우면 처리량(124/라운드)을 못 따라가 surplus 가 늙고 **age-cap flood** 가 난다(OPERATING_POINT §4.1 이 명시적으로 경고하는 *과적재 실패*). "큰 풀이 좋은가?"를 보려면 노드 풀이 아니라 **글로벌 후보 풀 richness(best-of-k)** 를 키워라(sim 무한풀 emulation).
3. **엣지 게이팅을 빠뜨리지 마라.** `gate()` 가 최장 ~2%를 edge 로 보내 interior 평균을 **워크로드 원래 평균(~116K) 과 무관하게 ctx_balance(100K)** 로 맞춘다. sim 은 cold_start *전에* gate() 적용 + streaming 프리필 arrival 도 게이트.
4. **`live_kv = prompt + 누적 decode`(매 라운드 +1).** steering 이 보는 건 prompt 가 아니라 이 footprint. 현실적 decode 길이는 **prompt-무관 짧은 분포**(예 uniform[256,4096])로 모델 — 옛 `dtot ∝ prompt` 는 긴 요청이 영영 안 끝나 live_kv 를 153K 로 부풀렸다.
5. **Σdev 의 의미**: 모든 배치의 `|Σkv−target|/target` *평균*(miss 만의 평균도, 최댓값도 아님). 거의 다 *명중한 정상 배치*가 차지(age_cap=∞·100% 명중서도 1.68%). **miss 는 전부 batch2**(남은 72→62 선택지 부족): count-miss(62 차기 전 KV 참) + dev-miss. 잉여는 오차를 *완화*하지 *유발*하지 않는다.
6. **Σdev 1.7% → 실제 idle ≈ 0**: PIM 이 GPU-A 밑에 숨어(t_pim ≤ t_gpuA) 흡수. 16%(최악 단일배치)는 드문 transient. 즉 현재 오차는 대부분 보이지 않는다.
7. **age-cap 5 = 공정성**(대기 ≤5 batch). starvation 0 보장. spread 0.7% knee(§3 sweep).
8. **`ctx_balance(100K)` 는 Llama70B+B200 도출값** — 다른 모델·HW 는 동작점 재도출(OPERATING_POINT §4). 특정 수치에 하드코딩하지 마라.

### 충실 검증값 (배포 동작점, cap5, best-of-2000, 다양 풀)
- **디코드: 명중 ≈99.5% / Σdev ≈1.7%** (안정: ITERS=4000·Z=128 → 99.53%/1.666%, early≈late 수렴).
- **프리필: 명중 100% / Σdev ≈0.1%** (이산 디코더 vs fungible 토큰 차이로 프리필이 훨씬 정밀).
- 상주분포 ≈20/70/10 보존, 풀평균 ≈100K, edge ≈1.93%.

---

## 3. 새 아이디어 구현 (임무 2단계)

`ideas/predictive_prepositioning_scheduler.md` 를 정독하라. 요지: **노드-로컬 steering 을 없애고, CPU 글로벌 스케줄러가 결정론적으로 각 노드의 2 배치(62+62)를 동작점에 맞춰 pre-position**. decode 완료는 `max_tokens`(=KV read 횟수)로 결정론적이라 빠질 것을 알고 *미리* 정확한 길이의 요청을 보낸다. 멀티턴=오프로드(복귀=새 긴 arrival), on-node 캐시(capacity 는 하드코딩 아니라 derive 산출 `hbm_capacity_tb − instance_a_tb`, 70B=1.33TB)에 mid·long 선택 캐시(방출=cache age-cap idle), 엣지 게이팅 유지. 동작점 age-cap 은 *사라지고*, 대신 **글로벌 풀 age-cap**(라우팅 FIFO 공정성)이 새 공정성 백본이 된다 — 둘 다 idea md 신설 절(「글로벌 풀 age-cap」·「capacity·방출·히트 측정」) 참조.

### 만들 것 (`ideas/proto/` 안에)
현재 `sim/lifecycle.cpp`(노드-로컬 steering) 를 **비교군**으로 두고, **글로벌 결정론 pre-positioning** 버전을 프로토타입으로 만들어 같은 워크로드에서 측정·대조하라:
- **멀티턴 워크로드 모델**: 요청에 `max_tokens`(결정론 decode 길이). 응답 종료 후 **길이-의존 복귀확률**(짧↓/길↑, 평균 ~2.5턴 = continue≈0.6/end≈0.4)로 다음 턴(이전 응답 누적된 더 긴 요청)으로 복귀 or 종료. 복귀까지 idle-age.
- **on-node KV 캐시**: capacity = derive 산출(`hbm_capacity_tb − instance_a_tb`, 70B=1.33TB·**하드코딩 금지**), mid·long 우선, **방출 = age-cap(idle)**. 히트=로드0 즉시 재개. reload 비용 = **상수 속도(offload BW)×KV 바이트(∝길이)** — 시간을 상수로 박지 말 것.
- **글로벌 pre-positioning**: 콜드스타트 124 정밀 구성(+엣지게이팅), 이후 prefill 완료·decode 종료를 예측해 정확한 길이 요청 pre-send. 노드는 실행만.
- **글로벌 풀 age-cap**: 라우팅이 길이-fit 만 보면 글로벌 풀 starvation → cap 이하서 강제 라우팅(FIFO), off-fit 은 글로벌 composer 가 흡수. 동작점 age-cap 빠진 자리의 주 공정성.
- ⚠ **유한 큐 모델 필수**: best-of-K(무기억 복원추출)는 도착 시각·정체성이 없어 FIFO/글로벌 age-cap 을 표현 못 한다. 도착 타임스탬프 달린 유한 큐로 전환(prepositioning 이 어차피 요청 identity 추적 필요).

### 측정·대조 지표
- Σdev(평균·최악·miss분류), p99 tail, edge 비율, **캐시 히트율(evict-age 함수·knee)**, **글로벌 age-cap↔Σdev knee**, **캐시 절약(시간/트래픽, reload=속도×길이)**, 100K 유지 여부 — **로컬 steering(현재) vs 글로벌 pre-position(신규)** 대조.
- 검증 질문(아이디어 md 의 "검증해야 할 것"): 복귀 타이밍 jitter 흡수에 필요한 *최소 슬랙*? 글로벌 제어 latency 가 forward-pass 박자를 따라가나? 캐시가 tail 을 얼마나 줄이나?

### 산출물
`ideas/proto/` 에 프로토타입 코드 + `ideas/proto/REPORT.md` 에 **실측 대조 결과**(빌드·실행해서 얻은 숫자). 추측 아닌 측정.

---

## 4. 맥락 메모
- 이전 세션 요약: lifecycle sim 의 힐링 버그(센터링) 발견·정정 → 충실값 ≈99.5%/1.7% 로 문서 수정·커밋(`e43a298`). 관측 진단(최악/miss/batch1·2 분류) 추가(`83052a1`). 멀티턴·pre-positioning 토론 → 이 아이디어 문서화.
- `ideas/` 전체가 gitignore(로컬). 프로토타입·리포트도 push 안 됨.
- 사용자는 길이 분산 무관·동작점 도출의 일반성을 중시한다. 특정 100K·134 같은 상수는 *Llama70B+B200 예시*이며 derive 로 일반화됨을 잊지 마라.
