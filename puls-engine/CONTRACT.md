# PULS-ENGINE — 통일 계약서 (CONTRACT)

> **이 문서가 단일 진실원이다.** 모든 모듈·에이전트는 여기 정의된 고정/변수 경계, 인터페이스,
> 네이밍, canonical 결정을 따른다. 코드와 이 문서가 어긋나면 **이 문서가 맞다** — 코드를 고친다.
> 근거 출처: `ARCHITECTURE.md`, `OPERATING_POINT.md`, 기존 `implementation/` 코드(레퍼런스),
> `implementation/analysis/*.cpp`(검증 sim). 기존 자산은 **읽기만** 했고 베끼지 않는다(문서 의도 기준 새 구현).

---

## 0. 목적 & 범위

특정 모델/하드웨어에 박힌 현재 구현을, **모델 스펙·GPU 스펙을 입력 변수로 받아 동작점을
스스로 도출하고 그에 따라 스케줄링하는, 하드코딩 없는 범용 C++ PULS 스케줄러**로 만든다.
출력값(100K·62·6.15M·128…)이 아니라 그 출력을 만드는 **공식·로직**을 코드화한다.

**공식 발명 금지** — 아키텍처/오퍼레이팅 문서 + 기존 코드에 *이미 있는* 로직만 옮긴다.

---

## 1. 고정 vs 변수 경계 (사용자 확정)

### 고정 (현 스펙 박제 — `core/spec.h::substrate`)
PULS를 정의하는 substrate. 모델/GPU가 바뀌어도 불변.
- **SP-PIM**: tile time(FP8 **267 ns**), tile rows(**32**), cross-GPU broadcast(0.5 ns), per-GPU stack 수(8), per-stack channel 수(32). 출처 config.py:123,303,304 / HWConfig.
- **HBM4 메모리**: BW 2.0 TB/s·stack, per-stack 채널수(32) 고정, 채널밀도 상한 16 Gb(JEDEC).
  **die-stack 높이는 변수** — 4-die SID 단위(4의 배수), 실용 12·16단. 용량은 JEDEC primitive
  에서 *산출*: `32ch × 채널밀도(16Gb@16단, die 선형) / 8 × stack수(num_gpus_a×8)`.
  → 16단·64스택 = **4.096 TB**, 12단 = **3.072 TB**. (문서 §4.1 "4.40 TB" 는 산술 오기 — 64GB×64=4.096.)
  `DeriveOptions.hbm_stack_height`(기본 16). 출처 JEDEC JESD270-4A.
- **KV 정밀도 = FP8(8비트)**: `KV_BYTES_PER_ELEM = 1`. FP16 경로 만들지 않음.

### 변수 (입력 — `core/spec.h::ModelSpec`, `HwSpec`)
- **모델**: num_layers, hidden, num_heads, num_kv_heads, head_dim, ffn_intermediate,
  **weight_bytes_per_elem**(가중치 정밀도 — FP8=1/FP16=2/FP32=4, 동적. KV 와 달리 *변수*).
- **GPU**: gpu_peak_tflops(dense FP16 per-GPU), gpu_mfu, num_gpus_a(TP, **기본 8**), num_gpus_b(TP, **기본 8**).
  (인스턴스당 8 GPU 는 배포 사실이라 기본값 8 — 단 *변수*로 둬서 측정 시 8 대입, 다른 구성도 산출 가능.)
- **knob**: prefill_tokens (배포 128 — *스케일 knob*, 동작점 basis를 정함).

> 함의: PIM/HBM/FP8를 고정해도 GPU·모델이 바뀌면 op-time 계수가 바뀌어 `ctx_balance`(현재 100K)·
> decode count·KV target·prefill scale이 **전부 다시 도출**된다. 그래서 산출 모듈이 필수다.

---

## 2. 세 산출물 분리 (코드 레벨 경계)

로직은 `core/` **하나**에 두고, 세 산출물은 그 위의 얇은 드라이버다. 같은 로직을 공유하되
경계를 코드로 분명히 둔다(검증이 통과해도 런타임이 다르게 도는 함정 방지).

| 산출물 | 디렉터리 | 역할 | 요청 출처(RequestSource) |
|---|---|---|---|
| **실 서빙 런타임** | `runtime/` | 실제 요청 라우팅·스케줄 결정 | 실 큐 (`QueueSource`) |
| **시뮬레이터** | `sim/` | 이산 동작 재현(라운드/churn) | 분포 B (`WorkloadSource`) |
| **검증** | `validation/` | 모듈별 + 통합 명중·Σdev·불변식 | 분포 B + 합성 fixture |

**RequestSource 추상화**가 셋을 가르는 단일 지점이다(§5.4). 분포 B·무한풀·확률 churn은
**sim/validation 전용** — `core/`의 런타임 경로에 절대 넣지 않는다.

---

## 3. 세 모듈 (core)

```
core/
  spec.h            # 고정 substrate 상수 + 변수 ModelSpec/HwSpec + op-time 1차 함수
  optime.h/.cpp     # op-time 모델 (PIM/FFN/GPU-A) — spec에서 산출. 산출/노드/sim 공용
  operating_point.h # OperatingPoint 구조체 (도출 결과)
  derive.h/.cpp     # ① 파라미터 산출 — 세 자원 균형 풀어 OperatingPoint 도출
  steering.h/.cpp   # 순수 steering(decode/prefill) + age-cap. 노드 공용 핵심
  request_source.h  # 요청 출처 추상 인터페이스 (runtime/sim 분기점)
  node_scheduler.h/.cpp    # ② 노드 스케줄러 — 풀 관리 + 센터링 admit + per-completion 힐링
  global_scheduler.h/.cpp  # ③ 글로벌 스케줄러 — 게이트 + cold-start + per-node 힐링 + on-point
  workload.h/.cpp   # 분포 B 샘플러 (sim/validation 전용 — 명시)
```

### ① 파라미터 산출 (`derive`)
입력(ModelSpec, HwSpec, prefill_tokens) → 출력(OperatingPoint).
**방법(문서가 손으로 한 것을 코드로):** `optime`의 PIM/FFN/GPU-A op-time을 같게 두는
**세 자원 균형을 직접 푼다**(t_PIM = t_FFN = t_GPU-A). 이 균형의 해가 `ctx_balance`,
`decode_count_target`, `kv_operating_target`이다.
- 문서의 `ctx_balance=(K2+1)/K1` 닫힌형은 이 균형의 *해석적 등가물*이며 K1·K2 대수가 문서에
  불완전하게만 적혀 있다(prose-only). 따라서 **닫힌형을 재구성하지 않고**, 코드에 실재하는
  op-time 공식으로 균형을 수치/대수로 푼다 = 공식 발명 0, 문서의 *방법*을 그대로 코드화.
- 도출 사슬(OPERATING_POINT §1·§2): prefill_tokens(knob) → 균형시간 X → FFN batch →
  N_dec = FFN_batch − prefill → kv_target = N_dec × ctx_balance → prefill_kv_work = prefill × ctx_balance.
- HBM 적합성(§4.1): decode 풀·prefill in-flight KV + 가중치 → Instance A 합(TB) ≤ HBM 용량
  (JEDEC 산출, 16단·64스택 = 4.096 TB; die-stack·num_gpus_a 에 비례).

### ② 노드 스케줄러 (`node_scheduler` + `steering`)
- **decode μ-batch steering**: `ideal=(kv_target−S)/(count_target−n)` closest-to-ideal 그리디,
  종료 `n<count_target ∧ S<kv_target`. 2 μ-batch는 **used 공유**(disjoint).
- **prefill steering**: prefill_tokens개를 depth-합 타깃으로 같은 그리디. depth = processed+chunk(+1).
- **age-cap**: `wait ≥ age_cap`이면 steering 무시 강제. **canonical: 가장 wait 큰 것 선택**(§4).
  선택분 wait=0, 미선택 wait+1. starvation ≤ age_cap+1 batch.
- **센터링 admit**: `ideal = ctx_balance×(cnt+1) − liveSum`로 풀 평균을 ctx_balance에 센터.
- **per-completion 힐링**: 완료 retire한 hole마다 `ideal = hole`로 like-for-like 되채움(toxic-fit).
  풀을 목표크기로 유지(ready 전이 우선 → 모자라면 admit). **batched(평균) 힐링은 금지**(긴 거 굶음).

### ③ 글로벌 스케줄러 (`global_scheduler`)
- **게이트(엣지 격리)**: 긴 것부터 shed, 남은 평균 ≤ ctx_balance + EDGE_BAND(1K).
- **cold-start 분배**: arrival순, `min|추가후 mean − ctx_balance|` 그리디, can_fit(cap ∧ count<NODE_MAX).
- **per-node 힐링**: §②의 per-completion(ideal=hole). inter-node swap 0, 무축출.
- **on-point 체크(검증용)**: disjoint K개 (count, kv±band) compose, 실패 롤백·중단.

---

## 4. Canonical 결정 (stale/불일치 해소)

추출에서 드러난 불일치를 **문서 의도 = 파이썬 production**으로 통일한다.

| 항목 | 채택(canonical) | 버림 | 근거 |
|---|---|---|---|
| **동작점 basis** | **도출** (모델+HW+knob에서 산출) | 256/128 하드코딩 | 일반화 목표. 검증은 Llama70B+B200+128로 62/6.15M 재현. |
| **age-cap 강제 대상** | **가장 wait 큰 요청** | cpp first-match break | OPERATING_POINT §3 "가장 오래 기다린 것", admission.py 의도. |
| **힐링 방식** | **per-completion (ideal=hole)** | batched (ideal=평균) | ARCHITECTURE §7.4 toxic-fit. batched는 명시적 대조 실패군. |
| **prefill depth** | **processed + chunk + 1** (다음 토큰) | +1 누락 | cluster_lifecycle.cpp 의미 정확. self-correcting이라 영향 미미하나 정확형 채택. |
| **prefill wait** | **decode와 분리된 prefill_wait** | 공유 wait | admission.py 분리 모델이 정밀. |
| **closest 탐색** | 정렬 + 이분(좌우비교) | 선형스캔 | 동치 결과, 성능상 이분 채택(런타임 지향). |
| **분포 B / 무한풀 / 확률 churn** | **sim/validation 전용** | runtime core 진입 | ARCHITECTURE §7.5 "honest disclosure" — 검증 가정. |

---

## 5. 인터페이스 계약 (헤더가 잠금)

### 5.1 네이밍 규약
- namespace: 전부 `puls`. 하위: `puls::substrate`(고정 상수).
- 타입: `PascalCase` (ModelSpec, OperatingPoint, NodeScheduler).
- 함수/변수: `snake_case` (derive_operating_point, ctx_balance, decode_count_target).
- 상수: `UPPER_SNAKE` (PIM_TILE_TIME_FP8_NS).
- 단위 접미사 필수: `_ns`, `_us`, `_tb`, `_tokens`, `_bytes`. 시간 내부 통일 = **마이크로초(us)**.
- 동작점 필드명은 config.py와 의미 일치: `decode_count_target`, `kv_operating_target`,
  `prefill_kv_work_target`, `age_cap`, `idle_band`(=0.10), `ctx_balance`(=문서의 ~100K).

### 5.2 핵심 타입 (spec.h / operating_point.h가 정의 — 변경 시 이 문서부터)
- `ModelSpec{num_layers,hidden,num_heads,num_kv_heads,head_dim,ffn_intermediate, weight_bytes_per_elem=2}`
- `HwSpec{gpu_peak_tflops, gpu_mfu, num_gpus_a=8, num_gpus_b=8}`  (GPU 수 = 변수, 기본 8)
- `OperatingPoint{ctx_balance, decode_count_target, kv_operating_target, prefill_tokens,
  prefill_kv_work_target, ffn_batch, balance_time_us, decode_pool, prefill_pool, age_cap,
  idle_band, node_max, node_min, node_footprint_cap, edge_band, instance_a_tb, hbm_fits}`
- `DeriveOptions{decode_surplus=25, prefill_pool=60, age_cap=5, idle_band=0.10,
  prefill_avg_depth_frac=0.56, node_max_surplus=10, edge_band_tokens=1000,
  footprint_headroom=1.22, hbm_stack_height=16}`
  - `decode_surplus=25` = **C 표준 동작점**(캐시 ON U-knee → decode_pool = 2×62+25 = **149** 상주, 재선택 자유도). `node_max_surplus=10` 은 글로벌 라우팅 목표(**node_max 134**)로 별개 — 노드가 decode_pool 149 까지 로컬 healing 으로 top-up(node_max=149 로 올리면 forced 폭발, 측정 확인).

### 5.3 op-time (optime.h) — 모든 모듈이 이걸로만 시간을 잰다
```
double t_pim_us   (long long sum_decode_kv_tokens, const HwSpec&);          // Σctx → PIM tiles
double t_ffn_us   (int batch_total, const ModelSpec&, const HwSpec&);       // FFN FLOPs
double t_gpu_a_us (int batch_total, long long prefill_attn_work_tokens,
                   const ModelSpec&, const HwSpec&);                        // QKV+O_PROJ+PREFILL_ATTN
```
정확한 FLOPs/타일 수식은 §6에 박제(추출 근거 줄번호 포함). 구현은 이 수식 그대로.

### 5.4 RequestSource (request_source.h) — runtime/sim 분기점
```
struct PulledRequest { int prompt; int dtot; };          // 길이-도메인(토큰 내용 없음)
struct RequestSource {
  virtual ~RequestSource() = default;
  // ideal 근방(cap 적합)에서 한 요청을 당김. 없으면 prompt<0.
  virtual PulledRequest pull_near(double ideal_tokens, long long cap_room_tokens) = 0;
};
```
- `WorkloadSource`(sim/validation): 분포 B + best-of-K로 ideal 근사 = 무한풀 emulation.
- `QueueSource`(runtime): 실제 대기 큐에서 ideal에 가장 가까운 실 요청 선택(없으면 prompt<0).

> 노드/글로벌 스케줄러의 admit·healing은 **RequestSource로만** 새 요청을 얻는다. 이로써
> 같은 스케줄링 로직이 runtime(실 큐)·sim(분포 B)에서 코드 공유되며 경계가 명확해진다.

---

## 6. op-time 수식 박제 (추출 근거 — 그대로 구현)

`peak_flops(n) = gpu_peak_tflops × 1e12 × gpu_mfu × n`  (config.py:215,269)

**PIM (Instance A attention)** — config.py pim_emulator:53-88
```
k_channels = num_gpus_a × substrate::HBM4_STACKS_PER_GPU × substrate::PIM_CHANNELS_PER_STACK   // 8×8×32=2048
tiles      = ceil( sum_decode_kv_tokens / (k_channels × substrate::PIM_TILE_ROWS) )            // rows=Σctx
t_pim_us   = (tiles × substrate::PIM_TILE_TIME_FP8_NS + substrate::PIM_BROADCAST_NS) / 1000
```
입력 = Σ decode KV **token 수**(개수·ctx 개별 아닌 합). batch dim 무관.

**FFN (Instance B)** — config.py:270-274
```
flops    = 6 × batch_total × hidden × ffn_intermediate
t_ffn_us = flops / peak_flops(num_gpus_b) × 1e6
```

**GPU-A (A proj + prefill-attn)** — config.py:221-238,249
```
qkv      = 2 × batch_total × hidden × (hidden + 2 × num_kv_heads × head_dim)
o_proj   = 2 × batch_total × hidden × hidden
prefill_attn = 2 × prefill_attn_work_tokens × hidden    // Σ(chunk×depth) = work_tokens
t_gpu_a_us = (qkv + o_proj + prefill_attn) / peak_flops(num_gpus_a) × 1e6
```

**KV bytes/token (FP8 aggregate)** — config.py:130-131
```
kv_bytes_per_token = 2(K,V) × num_kv_heads × head_dim × 1(FP8) × num_layers
```

**Instance A 가중치 (QKV/O proj, 동적 정밀도)** — OPERATING_POINT §4.1 (Llama70B ≈ 24 GB)
```
instance_a_weight_bytes = (hidden×(num_heads×head_dim + 2×num_kv_heads×head_dim)  // QKV
                         + hidden×hidden)                                          // O proj
                        × num_layers × weight_bytes_per_elem                       // 동적 FP8/16/32
```

**HBM 적합성** — OPERATING_POINT §4.1. KV(FP8) + 가중치(동적) 둘 다 64 스택을 점유.
```
instance_a_tokens = decode_pool × ctx_balance + prefill_pool × (prefill 평균 진행 depth)
                    // decode_pool = 2×N_dec+잉여 → 활성 2 μ-batch 가 들어감
instance_a_tb     = (instance_a_tokens × kv_bytes_per_token + instance_a_weight_bytes) / 1e12
hbm_capacity_tb   = (num_gpus_a×8 stack) × (32ch × 채널밀도(16Gb@16단, die 선형)/8) / 1000
                    // JEDEC 산출 — 16단·64스택 = 4.096 TB (문서 §4.1 "4.40" 은 오기)
hbm_fits          = instance_a_tb ≤ hbm_capacity_tb
```
> prefill 평균 진행 depth는 OP §4.1이 ~56K(=ctx_balance×0.56 근사)로 표기 — `DeriveOptions`에
> `prefill_avg_depth_frac=0.56`로 명시 노출(문서 수치, 추정 아님).

---

## 7. 검증 전략 (모듈별 + 통합 + 회귀)

`validation/`는 외부 의존 없는 헤더-온리 미니 assert(`test_framework.h`)로 짠다.

### 모듈별(단위) — 어디가 깨지는지 국소화
- `test_optime`: 박제 수식 ↔ Llama70B+B200 손계산 일치(±1%). PIM/FFN/GPU-A 각각.
- `test_derive`: (Llama70B, B200, prefill 128) → ctx_balance≈100K, decode_count≈62,
  kv_target≈6.15M, ffn_batch≈190, hbm_fits=true. prefill 256 → 123/12.3M(2×). **스케일 불변성**.
- `test_steering`: 분포 B 풀에서 (count, kv±band) 명중. heavy/short/bimodal 변종 무관 spread~1%.
  age-cap: 모든 길이 클래스 ≤ age_cap+1 batch 내 drain(starvation 0).
- `test_node_scheduler`: 센터링 admit 평균 → ctx_balance 수렴; per-completion 힐링 toxic-fit
  보존(긴 요청 비율 유지) vs batched 굶음(대조).
- `test_global_scheduler`: 게이트 edge%=f(E) 스케일불변; cold-start count∈[NODE_MIN,NODE_MAX]·
  on2 명중; healing drift 0(early≈late).

### 통합
- `test_integration`: derive→global(cold-start)→node(steering+힐링) 한 줄기. 동작점 명중·Σdev,
  inter-node swap 0 불변식.

### 메타/회귀
- `test_meta`: 모듈 인벤토리, 고정/변수 경계(고정 상수가 ModelSpec/HwSpec에 안 샘), CONTRACT의
  canonical 결정이 코드에 반영(batched 힐링 부재, age-cap=wait-max) 검사.
- 각 모듈 핀 고정 → 모델/HW 변수 변경·리팩토링 시 회귀로 국소화.

---

## 8. 빌드

- C++17, CMake. `core` static lib + `cluster` static lib + `runtime`/`sim` 실행파일 + `validation` ctest 타겟.
- 드라이버: `runtime`(실 서빙) · `sim`(콜드스타트) · `lifecycle`(프리필→디코드) · `csched`/`baseline`/`prepo`(클러스터 C/ablation/대조).
- 외부 의존 0(표준 라이브러리만). 결정론: 시드 고정 RNG(sim/validation만).
- 빌드 산출물은 `puls-engine/build/`(gitignore). `bash build.sh` → 14 ctest(core 9 + cluster 5) 통과.

---

## 9. 불변식 (전 모듈 공통)

1. 고정 substrate 상수는 `substrate` 네임스페이스에만. ModelSpec/HwSpec에 섞이면 위반.
2. 분포 B·best-of-K·확률 churn은 `workload`/`WorkloadSource`에만 — `core` 런타임 경로 금지.
3. 힐링은 per-completion(ideal=hole)만. batched 평균 힐링 코드 부재.
4. age-cap 강제 = 가장 wait 큰 요청.
5. 동작점 수치 리터럴(100000, 62, 6150000, 128…) 금지 — 전부 derive 산출/DeriveOptions.
   (substrate 고정 상수와 DeriveOptions 기본값만 리터럴 허용.)
6. inter-node swap 0, 완료 순간만 churn(무축출).

---

## 10. 흡수된 클러스터 레이어 (C — 글로벌 age-cap · 멀티턴 캐시 · 컨텐션 TBT)

연구 프로토타입에서 검증된 **승리 설계 C**를 엔진에 흡수한 레이어. `core` 노드 메커니즘(steering·admission·per-completion 힐링) 위에 **추가 모듈**로 얹히며, core 는 무수정 재사용한다.

- **모듈**:
  - `scheduler/queue.{h,cpp}` — `GlobalQueue`: 길이-fit 라우팅(`pull_near`) + **글로벌 age-cap 강제**(`pull_slot`: wait>cap 인 가장 오래된 것 강제 주입). 노드 age-cap(steering)과 별개의 *클러스터* 공정성.
  - `scheduler/cache.{h,cpp}` — `ClusterCache`: **3-tier 멀티턴 KV 캐시**(HBM hit / SSD reload / recompute). 적격 = `len > eligibility` ∧ 노드 잔여 HBM. `evict_age` idle → SSD 강등, `gone_age` → 소멸. **`peek`**(부작용 없는 조회 — 어피니티 라우팅 판단) + **`enforce_budget`**(노드별 동적 예산 — 멀티턴 인플레이션으로 풀 실 KV 가 설계 footprint 를 초과한 만큼 캐시 차감, HBM 바이트 폐루프).
  - `scheduler/preposition.{h,cpp}` — pre-position(대조군 B 전용).
  - `sim/{harness,kpi,metrics}.h` · `sim/workload_mt.{h,cpp}` — **컨텐션·의존성 TBT**(`instance_a_latency = max(t_pim,t_gpu_a)+β·max(0,t_pim−t_gpu_a)`, `TBT = max(instance_a,t_ffn)×layers`) + 실 KPI(TBT/TTFT/SLO goodput/PIM-노출) + 멀티턴 워크로드(`max_tokens=uniform[256,4096]`).
  - `sim/{csched,baseline,prepo}.cpp` — 드라이버 C(채택)/A(잉여25 ablation)/B(pre-position 대조). C 는 **캐시 어피니티**(HBM-hit 복귀 → 보유 노드 전용 대기열, hole 발생 시 1순위 admit — 길이·cap_room 무관 최고령 우선; 대기열 비었을 때만 2순위 `pull_slot`) + **어피니티 spill cap**(대기 > cap 이면 글로벌 큐 spill — 글로벌 cap 과 분리: 캐시 보유 복귀는 대기의 보상이 있어 경제학이 다름) + **[PAUSE] KPI**(요청별 토큰 간격 — 잉여 일시정지 비용은 배치-단위 TBT 의 사각; node age-cap 이 최악 gap 을 cap+1 라운드로 bound) 포함.
  - `validation/test_{queue,cache,kpi,preposition,workload_mt}.cpp` — 큐 강제·3-tier 비용·컨텐션 TBT·pre-position·멀티턴 검증.
- **C 표준 동작점(확정, 2026-06 갱신)**: `decode_surplus=25`(→ decode_pool 149, §5.2 DeriveOptions) · `prefill_pool=80`(2-active 라운드당 2×128 요청-disjoint 정합 — 배치1 그리디의 깊이 선점으로 옛 60 은 배치2 96.3%/1.84% 미달, 80 = knee; Instance A 3.02→3.20 TB, 12단 미적합·배포 16단 적합) · `global_age_cap=25` · `eligibility=16000` · `evict_age=200` · `contention β=0.5` · `offload_bw=1e8`(현실 NVMe 어레이 — 옛 2e7 은 recompute 손익분기 2.1e7 아래라 티어 2 무력) · `node_age_cap=5` · `affinity=on` · `dyncache=on` · `aff_spill=200`. 재현: `csched 8000 64 16000 200 25 300 0.5 1e8 5 25` → Σdev 1.419% · hbmHit 87.9%(**물리 99.7%**) · SLO goodput 3.79M · TTFT 0.78M · max_wait 26 · 최악 토큰 간격 6 라운드(= node age-cap+1) · 캐시 예산 0.891 TB. `puls_lifecycle 4000 64 5 2000` → decode 99.92%/1.280% · prefill 99.5%/0.32%(b1 100%/0.107 · b2 99.06%/0.537) · 전이 1286(단일 670 의 ~1.9×).
- **불변식 준수**: 동작점은 여전히 `derive_operating_point`(harness `make_deployment`)로 산출 — 모델/HW 바뀌면 자동 재도출(§9-5, 동작점 리터럴 0 확인). 클러스터 knob(eligibility·β·offload_bw·evict_age)은 *가정 라벨 sim 파라미터*로 `sim`/`scheduler` 레이어에만(§9-2 워크로드 파라미터와 동격), `core` 런타임 경로 불침투.
- **미모델링(deferred)**: 이벤트-구동 디스패치 DAG(ARCH §6.3) — TBT 는 해석적 정상상태(더블버퍼링). 스케줄러 로직 재사용, 타이밍 층만 교체 시 도입.
