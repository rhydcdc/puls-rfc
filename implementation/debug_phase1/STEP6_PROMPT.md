# STEP 6 작업 프롬프트 (새 대화용) — 밸런스를 PIM-시간 기준으로 + TBT 측정

아래 내용을 새 대화창에 그대로 붙여넣으세요.

---

PULS 스케줄러 디버깅의 STEP 6 을 진행한다. 핵심 주제: **밸런스 게이트 기준을
"유휴율" → "PIM 시간 vs GPU 시간"으로 바꾸고, 지표를 idle → TBT 로 전환**한다.
수정 전에 아래 문서·코드를 **꼼꼼히 다 읽고** 맥락을 완전히 파악한 뒤 시작해라.
맨 아래 작업 습관 당부를 반드시 지켜라(이번 세션에 같은 실수 여러 번 했다).

## 먼저 읽을 것 (순서대로)

1. `README.md` / `ARCHITECTURE.md` — PULS 아키텍처 전체. 특히 ARCH **§5.3(compute-bound
   window 중 PIM overlap), §5.6(intra-instance double-buffering = F2), §6.3(PIM completion
   computed at dispatch), §6.4(adaptive admission)**. F2 의 본질 = "한 자원의 일을 다른
   자원의 더 긴 시간 *그늘에* 숨겨 cycle 을 안 늘리고 throughput 을 얻음".
2. `배치_생애.md` (repo 최상위) — 배치 생애 설계. **세 한계 + per-mb KV 예산(네 번째 강제)**,
   양방향 합류, 게이트, 종료. **밸런스의 진짜 정의를 STEP 6 에서 이 문서에 다시 써야 함**
   (현재는 "유휴율" 뉘앙스 → "prefill 을 PIM 시간 그늘에 숨겨 TBT 보존"으로).
3. `implementation/debug_phase1/PLAN.md` — STEP 1~5.5 체크리스트.
4. `implementation/debug_phase1/REPORT_baseline.md` — **§12(race·per-mb 예산), §13(ns/µs
   단위 버그), §14(스케일 스펙트럼)** 이 이번 세션 핵심. 반드시 정독.
5. 핵심 소스: `implementation/src/puls_sched/` 의
   - `admission.py` — `balance_pim_slack`(시간 기준 chunk 산식, **여기가 STEP 6 핵심**),
     `balance_intra_A`(유휴율 기준), `balance_inter_AB`, `layer1`.
   - `main_loop.py` — `_make_t_pim_fn`(ns→µs 수정됨), `_compose_admission_payload`(t_proj·
     t_pim_fn·per_token 산출), `_recompose_mb`(매 cycle 재구성, **budget freeze 지점**),
     `_try_join`(유휴율 게이트), `_per_mb_kv_budget`(=KV캐파/`_STAGGERING_TARGET_MB`=2).
   - `dispatcher.py` — `_op_time`(GPU=초×1e6 µs, **PIM=op_time(ns)×1e-3** µs). cycle 구조.
   - `pim_emulator.py` — `op_time` 은 **ns 반환**(tile_time_ns 기반, sequence-parallel 2048ch).
   - `config.py` — `AdmissionConfig`(prefill_chunk_default=512, idle_theta_high=0.3,
     idle_theta_low=0.05, max_batch_size=256), `compute_gpu_op_time_s`(per-op 초 산출).

## 이번 세션(STEP 5)에서 밝혀진 것 — 반드시 숙지

### A. PIM 유휴는 워크로드·물리 함수지 버그가 아니다 (지표가 틀렸었다)

- PIM(decode-attn)은 2048채널 sequence-parallel 이라 **고ctx 에서도 빠르다** (ctx 70K,
  27요청 decode-attn ≈ **7.74µs**). GPU projection(QKV+O_PROJ) ≈ **6.18µs** →
  순수 decode 에선 **PIM/GPU = ctx/56,160 = 1.25 (PIM-bound)**.
- 그러나 **PREFILL_ATTN = O(chunk×ctx)** 라 고ctx 에서 폭발적 (3요청·chunk 170·ctx 70K ≈
  **444µs**, PIM 의 57배). → **prefill 이 조금이라도 있으면 GPU 가 압도 → PIM 유휴.**
- 결론: **PIM-bound(PIM 바쁨)는 오직 `ctx>56K + 순수 decode(prefill 없음)`** 한 구간뿐.
  나머지는 GPU-bound·PIM 유휴가 *정상*. "PIM 유휴율 낮추기"는 애초에 틀린 목표였다.

### B. 스케일 스펙트럼 (수정된 코드, 엄밀판 측정, 전부 mb=3·window 3/3)

```
              gpu_a idle  pim_a idle   ctx       해석
T-S ON:        0.0002      0.9943      ~8K       저ctx → GPU-bound (PIM 일감 微, 정상)
T-GEN ON:      0.0002      0.9935      ~13K      저ctx → GPU-bound
T-GEN OFF:     0.0010      0.9913      ~13K      OFF≈ON (저ctx 합류 무관)
agentic OFF:   0.343       0.458       ~70K+     순수 decode → PIM 활용(유휴 46%)! 둘 다 ~60%
agentic ON:    0.002       0.977       ~70K+     합류가 GPU 를 prefill 로 채움
```
- agentic OFF(둘 다 ~60% 활용) = **balance/staggering 이 만든 F2 regime** (PIM 이 한 mb
  decode-attn 하는 동안 GPU 가 다른 mb projection overlap). 메커니즘은 작동한다.
- agentic ON = 합류가 GPU 유휴(34%)를 backlog prefill 로 채움. **하지만 이게 문제다(아래).**

### C. ★ 핵심 문제 — 합류의 GPU 충전이 TBT 를 폭증시킨다

cycle = `t_qkv + max(t_prefill_attn, t_pim) + t_oproj`. 고ctx 에서:
```
OFF (prefill 없음):  cycle ≈ max(6µs, 7.74µs) = 7.74µs   → TBT ≈ 7.74µs × 80층 ≈ 0.6 ms/token
ON  (prefill flood): cycle ≈ 6 + 444 + ... ≈ 567µs       → TBT ≈ 567µs × 80 ≈ 45 ms/token (~75×!)
```
- GPU 100% 활용은 **좋은 게 아니라** prefill 이 PIM 슬랙을 넘어 cycle 을 늘려
  **decoder 의 TBT 를 파괴**한 것. (mixed batch 라 decoder 가 같은 cycle 의 prefill 뒤에서
  대기.) PIM 은 7.74µs 에 끝내고 GPU 가 prefill 444µs 가는 동안 논다 = 그 대기가 TBT.

### D. ★ STEP 6 의 본질 — 게이트 기준을 유휴율 → PIM 시간으로

밸런스의 *진짜* 정의(F2): **한 사이클에서 t_pim 과 t_gpu 를 재서, PIM 이 더 길면 그
슬랙(t_pim − t_gpu)만큼만 GPU 에 prefill 을 끼워 cycle 을 안 늘리고(=TBT 보존) throughput
획득. GPU 가 더 길면 그 슬랙에 decode-attn(싸다)을 더 채워 batch↑**. 유휴율이 아니라
*시간*이 기준이어야 한다 (ARCH §5.3/§5.6).

현 코드가 어긋난 3 지점:
1. **`balance_pim_slack` 의 `max(base 512, chunk_optimal)` floor** — 고ctx 에서 optimal≈1
   인데 base 512 가 덮어써 슬랙 초과 → TBT 폭증. **floor 제거/재설계 필요.**
2. **`mb.prefill_chunk_budget` 가 admission 시점에 한 번 산출 후 freeze** — in-flight mb 는
   매 사이클 t_pim 이 변해도 재계산 안 함(`_recompose_mb`). **매 recompose 재계산 필요.**
3. **합류 게이트(`_try_join`: gpu_idle>θ OR pim_idle>θ)·`balance_intra_A`(gpu_idle>θ_high)
   가 유휴율 기준** — "한 사이클 prefill 양"을 누적 유휴%로 결정. **양은 시간 슬랙으로,
   유휴율은 "언제 admit/합류할지(용량 게이트)"에만** 쓰도록 분리.

(주의: 단위 버그(§13)는 이미 고쳤으나 agentic 측정엔 무효였다 — t_proj 가 커서 슬랙이
음수 → 어느 단위든 chunk_optimal=0 → base 512. 즉 floor 가 진짜 범인. 단위 수정은
양의 슬랙 regime 위해 유지.)

### D-2. ★★ 멤버십(용량) vs 밸런스(시간) 분리 — 유휴율을 *기준*에서 제거

스케줄러가 정하는 건 사실 **두 개이고 서로 다른 축**인데 현 코드가 유휴율로 섞어놨다:

- **(1) 멤버십 = "어떤 요청을 배치에 넣을까"** → **용량 함수**: seq cap(256) + KV 캐파 +
  per-mb KV 예산 + 큐 비었나. 자리 나면 채움 = 연속배칭. **유휴율 불필요.** ("놀까봐
  가져온다"가 아니라 "자리 나서 채운다".)
- **(2) 사이클 내 일 배분 = "한 사이클에 prefill 얼마"** → **시간 함수**: PIM 슬랙
  (t_pim − t_gpu)만큼만 → cycle 안 늘림 → **TBT 보존**. 이게 밸런스.

**유휴율은 input(기준)이 아니라 output(결과)이어야 한다.** 시간 균형(t_pim≈t_gpu)을
맞추면 유휴율은 그 *결과로* 자연히 낮아진다. 유휴율을 *기준*으로 두면 "놀까봐 GPU 에
계속 prefill 끼워넣는" 반응 피드백 → 과주입 → TBT 폭증(C 절).

→ **할 일**: 유휴율을 *판단 기준*에서 제거.
- `_try_join` 게이트(`gpu_idle>θ OR pim_idle>θ`) → **용량 기준으로**(자리+큐). hysteresis
  (θ_low/high)도 게이트 제거되면 moot.
- `balance_intra_A`(`gpu_idle>θ_high` → chunk += n_sat) → **시간 기준 prefill 사이징으로 흡수.**
- **종료** = "큐 빔 + in-flight 완료"(용량/큐 기준)로 자연 처리 — 게이트 닫힘에 의존 X.
- **`idle_telemetry` 자체는 유지** — 측정/진단/보고용(시간 균형 잘 됐는지 *관측*). 기준 아님.
- 착수 시 **`grep -rn idle src/` 로 모든 사용처 확인** 후 기준-용도만 제거(측정-용도 보존),
  종료·evict 로직이 게이트에 의존하지 않는지 검증.

핵심: **요청을 많이 admit 해도(용량 허용) 사이클당 prefill 은 PIM 슬랙으로 제한되니
TBT 보존.** 멤버십과 사이클 일을 분리하면 "많이 담되 천천히 prefill"(TTFT↔TBT trade-off는
별도 정책 knob 판단).

## STEP 6 에서 할 일 (측정부터, 추측 금지)

> 이번 세션에 가설이 5번 빗나갔다. **반드시 측정으로 확인**하며 진행해라.

1. **measure_steady 에 TBT 산출 추가** — decode 토큰당 평균 cycle 시간
   (= 측정 구간 clock_span / 그 구간 생성된 decode 토큰 수, 또는 per-request 평균).
   idle 은 보조로 남기되 **주 지표 = TBT**.
2. **현 코드 OFF/ON TBT 측정** (agentic) → "ON TBT ~75× 악화" 가설 실측 검증.
3. **시간 기준 balance 로 수정**:
   - prefill 양 = max(0, t_pim − t_gpu)/per_token, **base floor 제거**(슬랙 0 → prefill 0 =
     순수 decode → TBT 최소). 단 TTFT(신규 prefill 진행) 와의 trade-off 고려 — 최소
     보장량을 둘지 설계 판단.
   - **매 `_recompose_mb` 에서 현재 t_pim/t_gpu 로 재계산**(freeze 해제).
   - 유휴율 게이트는 admit/합류 *시점*에만, *양*은 시간 슬랙으로.
   - 대칭: GPU-bound 면 decode 를 PIM 슬랙에 더(decode-attn 싸서 GPU 그늘에 숨음).
   - 단위테스트 + 회귀 동반.
4. **TBT 재측정** → 시간 기준 balance 가 TBT 보존하며 throughput 얻는지 확인.
5. **문서 갱신**: 배치_생애 §밸런스 정의(유휴율→PIM 시간 그늘), README "Target Workload"
   재프레이밍(주 타깃 = **long-context agentic**: 큰 prefill + 큰 decode; PIM 가치는
   가동률 아니라 op-level Aux2·F5), REPORT §15.

## 중요한 사전 지식 (헷갈리기 쉬운 것)

- **임계 56,160**: 순수 decode 의 t_pim/t_gpu_proj = ctx/56,160. ctx>56K 면 PIM-bound.
- **단위**: `PIMExecutor.op_time` = **ns**. clock·GPU op_time = **µs**. dispatcher 가 PIM 을
  ×1e-3, `_make_t_pim_fn` 도 ×1e-3(수정 완료). 새 코드에서 시간 비교 시 단위 맞춰라.
- **타깃 워크로드 = long-context agentic** (사용자 확정): 긴 컨텍스트 읽고(prefill 큼) +
  긴 추론 생성(decode 큼). causal 비대칭(prefill-attn 은 평균 절반 ctx, decode-attn 은
  full ctx)으로 **decode ≈ prompt 절반이면 decode-attn 일이 prefill-attn 에 맞먹음**.
- **PIM 가동률은 목표가 아니다** — 적은 decode-attn 을 *얼마나 싸게*(버스 절감 Aux2,
  per-token TBT) 하느냐가 가치. 저/중 ctx 에선 PIM 유휴가 정상.
- **합류는 연속배칭(throughput) / 밸런스는 cycle 내 일 배분(TBT)** — 직교하나 현재
  합류가 유휴 보고 prefill 을 과주입해 TBT 를 깬다.

## 측정 운영 (`measure_steady.py`)

- 엄밀판: 워밍업(decode 진입까지 `--warmup-decode-frac`, 기본 0.5; 순수 decode 보려면
  0.9) → `idle_telemetry.reset` → 수렴(Δ<0.005) 정지. 완주 안 함.
- 인자: `--trace --label --batch --theta-high --theta-low --chunk --no-join
  --warmup-decode-frac`. config 무수정 override.
- 트레이스: `data/trace_agentic.csv`(고ctx 장기추론, 주력), `trace_general.csv`(일반),
  `trace_ts.csv`(short), `trace_tdec/tm.csv`. 생성: `gen_agentic.py`·`gen_general.py`·
  `gen_step5_traces.py`.
- 진단: `diag_optime.py`(op_time 직접 산출 — 단, 출력에 ns/µs 주의), `diag_join_race.py`.
- 실행 예: `cd implementation && PYTHONIOENCODING=utf-8 python debug_phase1/measure_steady.py
  --trace debug_phase1/data/trace_agentic.csv --label X --warmup-decode-frac 0.9`

## 현재 상태 (커밋 4개, 미푸시)

```
3bcbaaa docs(phase1): STEP 5 scale-spectrum measurements + findings
1a1feb5 fix(phase1): correct ns/us unit in balance t_pim_fn
b9f3ffc feat(phase1): per-mb KV budget to restore F2 μ-batch staggering
9368edf feat(phase1): revive idle_theta_low as join-gate hysteresis deadband
```
- working tree 깨끗 (untracked: `analysis/`, `_scratch_*` 무관).
- **4 커밋 origin/main 미푸시** — 필요시 푸시 먼저.
- 테스트: prefill_join 13, admission+chunk+payload 68+12, 빠른 정합성 61, lifecycle+e2e 18
  — 전부 passed. **풀 회귀는 STEP 6 커밋 게이트에서 1회.**

## 작업 습관 당부 (이번 세션 실수 — 꼭 지켜라)

- **모든 bash 명령에 `cd /c/Users/rhs02/Desktop/puls-rfc/implementation &&` 붙여라.**
  cwd 가 안정적으로 유지 안 됨 — 이번에 5번쯤 빠뜨려 크래시/엉뚱한 디렉터리.
- **`run_in_background` 안에서 `&` 쓰지 마라** — detach 되어 완료 알림이 깨진다.
- **추측하지 말고 측정해라.** PIM 유휴 원인·decode 지배·op_time 등 **가설이 5번 빗나갔다.**
  op_time 같은 건 `diag_optime` 로 직접 산출해 확인(단위 ns/µs 조심).
- **한 메시지에 도구 과병렬 금지** (cascade 취소). 무거운 측정은 백그라운드 하나씩.
- **변경하면 바로 커밋.** 회귀는 개발 중 가벼운 타깃만, 커밋 직전 풀 1회.
- **PYTHONIOENCODING=utf-8 + 파일 출력** (콘솔 cp949 유니코드 깨짐). 커밋 메시지는 bash heredoc.
- **PowerShell here-string·`->`·유니코드 깨짐 주의.**
- 수정 후 의도 정합을 **배치_생애.md / ARCH §5.3·§5.6 기준**으로 자가검증.

시작 전 위 문서들 다 읽고, **(1) TBT 측정 추가 → (2) 현 OFF/ON TBT 실측 → (3) 시간 기준
balance 수정 설계**의 접근 계획을 먼저 제시해라.
