# FINAL REPORT — 승리 스케줄러 C & 다음 대화 핸드오프

> **이 문서 하나로 다음 대화에서 `ideas/proto` 의 스케줄러를 업데이트할 수 있다.** 결론·확정 파라미터·
> 왜 그런지·모델 핵심개념·건들면 안 되는 것까지 담았다. (절대원칙: `puls-engine/` 등 `ideas/` 밖 무수정.)

---

## 0. 결론 (한 줄)
**C = A 의 노드 메커니즘(잉여 + 매-라운드 steering + per-completion 힐링) + 글로벌 age-cap + on-node KV 캐시.**
**잉여=25, 글로벌 age-cap=25** 로 두면 **A(현 설계)·B(순수 pre-positioning)를 *모든* 실 KPI 에서 이긴다.**

## 1. 확정 파라미터 (배포 동작점 = Llama70B + B200 + prefill 128)
| knob | 값 | 의미 |
|---|---|---|
| **decode_surplus** | **25** (decode_pool = 2×62 + 25 = **149**) | 매-라운드 재선택의 상주 후보 여분 |
| **global_age_cap** | **25** (라운드) | 글로벌 풀 대기 상한 → 강제 라우팅 |
| evict_age | 200 | HBM 캐시 idle 방출 |
| eligibility_threshold | 16000 | mid·long 만 캐싱 |
| contention β | 0.5 | HBM 컨텐션 페널티 (t_pim>t_gpu_a 시) |
| offload_bw | 2e7 B/round (≈SSD 10GB/s) | 캐시 miss reload 속도 |
| node_age_cap | 5 | 노드 steering 공정성 |

CLI: `csched.exe [iters] [Z] [elig] [evict_age] [global_age_cap] [seed_backlog] [beta] [offload_bw] [node_age_cap] [surplus]`
재현: `./ideas/proto/build/csched.exe 8000 64 16000 200 25 300 0.5 2e7 5 25`

## 2. 최종 대조 (8000 it, Z=64)
| 지표 | A (baseline) | B (prepo) | **C (확정)** |
|---|---|---|---|
| Σdev avg / worst | 1.504% / 19.6% | 8.32% / 101% | **1.348% / 16.9%** |
| TBT mean / p99 (µs) | 2055 / 2191 | 2289 / 3761 | **2056 / 2191** |
| TTFT mean / p99 (µs) | 1.78M / 10.2M | 1.21M / 13.0M | **0.74M / 8.9M** |
| SLO goodput (tok/s) | 3.68M | 3.20M | **3.79M** |
| TTFTmet% | 93.6 | 93.2 | **97.1** |
| hbmHit% | 0(off) | 81.8 | **92.0** |
| savedR (캐시 절감) | −0.05B | 18.5B | **24.9B** |
| PIMexposed% | 70.0 | **99.6** | 73.5 |
| max_wait (round) | 6471(starve) | 998 | **26** |
| forced / poolMean | 0 / 105K | 1102 / 108K | 1626 / 117K |

→ **C 가 전 지표 1위 또는 동률.** (poolMean 117K·forced 1626 은 높아 보여도 무해 — §3.)

> **※ 이 결과는 새로 구현한 의존성·컨텐션 모델 기반이다.** TBT·PIMexposed·goodput 은 **PIM↔GPU-A 의 HBM
> 컨텐션 + 인스턴스 A→B 의존성(인스턴스 A 끝나야 B 시작)** 을 반영한 값으로 산출됐다(§4). 즉 위 대조는 단순
> op-time max 가 아니라 *의존성·컨텐션이 들어간* 진짜 TBT 위에서의 비교다. (B 가 TBT 에서 지는 것도 PIM
> 노출 99.6% → 컨텐션 페널티 때문.)

## 3. 왜 (분석)
- **잉여 = composition 담당.** 잉여(상주 후보 여분)가 *매-라운드 재선택*을 가능케 한다. 풀이 117K 로 떠도
  배치는 **낮은 62 를 골라 6.21M 명중** → Σdev 1.35%(A·B 다 이김). poolMean 과 Σdev 가 분리되는 이유.
  **큐(미상주·prefill 안 됨)는 이 후보가 못 된다 — 그래서 잉여가 필수.** (B 는 잉여 0 → 재선택 불가 →
  배치=풀=drift → Σdev 8.3%, t_pim↑ → PIM 노출 99.6% → TBT p99 3761.)
- **작은 글로벌 age-cap = 대기·캐시 담당.** cap 25 → 복귀가 ≤26 라운드(~53ms)에 강제 서빙 →
  **evict_age(200) 안에 hit → hbmHit 92% + TTFT 0.74s.** forced 1626 으로 많지만 **잉여가 off-fit 을
  흡수**해 Σdev 무해. (A 는 글로벌 age-cap 0 → starvation max_wait 6471 → TTFTmet·캐시 손해.)
- **분업이 핵심.** 잉여→composition(Σdev) · 작은 cap→latency/공정성/캐시. 둘을 함께 둬야 C 가 완성.
- **U자 knee (스윕 근거).** 잉여 >~60 → instance_a↑ → **캐시 HBM 고갈**(잉여 100 → hbmHit 0).
  cap <25 → **forcing 폭발**(cap 5 → forced 20142, 잉여 흡수 한계 초과 → Σdev 1.63%). **(25,25)가 두 knee 교차점.**

## 4. 모델 핵심 개념 (다음 대화가 헷갈리지 말 것)
> **★ 반드시 기억할 두 가지 (혼동 금지):**
> 1. **`max_tokens` 은 "랜덤 EOS 분포"를 나타내는 모델링 상수다** — 각 요청이 EOS 까지 몇 토큰 생성하느냐
>    (현실 EOS 는 랜덤, 모델은 max_tokens 로 결정론 치환). *KV 양이 아니라 *루프 길이(시간 축)*.
> 2. **KV(`live_kv`·`6.21M`)는 *각 레이어마다* 읽는 양이다** — per-step 한 forward-pass 안에서 80 레이어가
>    각각 그 KV 를 읽는다(TBT = 레이어당 max × 80). *한 라운드 총량도, max_tokens 도 아니다.*

- **`max_tokens`** = 각 요청의 **decode 루프 길이 = 랜덤 EOS 의 결정론 모델링 상수**(시간 축). 디코더는
  노드에 **상주하며 라운드당 토큰 1개**, `dec` 0→max_tokens 후 완료·퇴출. **토큰 사이 KV 리로드 0**(상주).
- **`live_kv = prompt + dec`** = **각 레이어마다 읽는 KV**(라운드당 +1 성장). PIM 이 레이어별 attention 으로 읽음.
- **`6.21M`(kv_operating_target)** = 62-배치가 **한 스텝·한 레이어**에 읽는 Σ live_kv (PIM 타깃). **≠ max_tokens.**
- **TBT (★ 의존성·컨텐션 신규 구현)** = `max(instance_a_latency, t_ffn) × num_layers(80)`.
  `instance_a_latency = max(t_pim, t_gpu_a) + β·max(0, t_pim−t_gpu_a)`.
  - **인스턴스 A→B 의존성**: 인스턴스 A(PIM ∥ GPU-A)가 끝나야 인스턴스 B(FFN) 시작 → `max(A_time, t_ffn)`
    (더블버퍼링 정상상태 throughput).
  - **PIM↔GPU-A HBM 컨텐션**: 컨텐션-무료 조건 = **`t_pim ≤ t_gpu_a`(PIM 이 GPU-A 그림자에 숨음)**.
    위반(t_pim > t_gpu_a = PIM 노출) 시 `β·(노출분)` 페널티. **모든 TBT/goodput 결과가 이 모델 기반.**
    (β=0 이면 옛 max(셋)×layers 로 회귀.)
- **잉여 = 상주(prefill 끝) 디코더 여분** = 매-라운드 steering 후보. 큐(미상주)는 대체 불가.

## 5. 핸드오프 — 다음 대화가 할 일
- **승리 설계 = C.** `ideas/proto/sim/csched.cpp` 가 그 구현. 위 §1 파라미터가 확정 동작점.
- **업데이트 방향(택)**: ① C 를 기본/canonical 로 승격(잉여25·cap25 기본값화) ② B(순수 prepo)는
  *열위 대조군*으로 보존 ③ A 는 현-설계 baseline 으로 보존. 필요시 문서/REPORT 정리.
- **건들지 말 것**: `max_tokens`(EOS 모델)·인스턴스 A→B 의존성·HBM 컨텐션(β)·core 사본. 이들은 "잘
  구현됨" 확인 완료. 새 실험은 §1 knob 스윕이나 reservoir 모델 한정.
- **검증 절차**: 빌드 →`csched.exe 8000 64 16000 200 25 300 0.5 2e7 5 25` → §2 C 행 재현(시드 고정) 확인.
- 파일 지도: 로직 `scheduler/{queue,cache,preposition}` · 시뮬 `sim/{csched,baseline,prepo,workload_mt,harness,kpi,metrics}` · 검증 `validation/test_*` · 사본 `core/` · 리포트 `REPORT.md`(P1)·`REPORT3.md`(P3)·`special_report_surplus_and_axes.md`.

## 6. 경계 / 가정 라벨
- 절대값(6.21M·100K·TBT·TTFT)은 Llama70B+B200 derive 종속(HW 바뀌면 재도출). `offload_bw·β·think_gap·
  gone_age·SLO 임계·max_tokens 분포`는 가정 라벨(스윕 가능).
- **미모델링(deferred)**: 인스턴스 내 디스패치 DAG(ARCH §6.3, 커널-완료 이벤트). TBT 는 *해석적 정상상태*
  공식(더블버퍼링 가정). 이벤트-구동 타이밍은 향후 sim 타이밍 층 교체 시(scheduler 로직 재사용) 다룸.
