# STEP 6 작업 프롬프트 (새 대화용) — ★ 근본 재설계: persistent-mb → 풀(pool) 모델

아래 내용을 새 대화창에 그대로 붙여넣으세요.

---

PULS 스케줄러의 **근본 재설계**를 진행한다. 결론부터: 지금까지의 Phase-1 디버깅
(STEP 1~5.5)에서 고쳐온 문제들(단일 mb 독점, 합류 cannibalization, per-mb KV 예산,
유휴율 게이트, TBT 폭증)은 **전부 "요청을 micro-batch 컨테이너에 생애째 가둔" 모델의
부작용**이었다. 올바른 모델은 **전역 풀 + 매 iteration 혼합 배치 선택(Sarathi-Serve 식)**
이고, 이게 PULS 가 *원래 표방한* 설계다(ARCH §6.1 "phase mix", README "Sarathi 의 PIM
확장"). **persistent-mb 코드를 더 재거나 고치지 말고, 풀 모델로 코어를 재설계한 뒤
측정한다.**

> 시작 전 아래 문서·코드를 꼼꼼히 다 읽고, **맨 먼저 이 결론을 ARCH §6 정독으로
> 검증**해라(맹신 금지 — μ-batch window 가 persistent 의도였는지 per-iteration 의도였는지).
> 맨 아래 작업 습관 당부를 반드시 지켜라.

## 0. 가장 먼저 — 이 재설계 방향을 검증하라

이전 세션이 도달한 결론이나, **새 세션은 추측을 검증부터 한다**(가설이 여러 번 빗나갔다):
- **ARCH §6 전체(특히 §6.1 μ-batch Composition, §6.2 DAG, §6.3 dispatch, §5.6
  double-buffering)를 정독** → "μ-batch = 풀에서 매 iteration 선택한 혼합 배치"가
  맞는지, 아니면 persistent 컨테이너 의도였는지 확정.
- 핵심 인용 (line 330): *"A μ-batch contains **different requests in a phase mix**"* /
  (line 275) *"prefill and decode **coexist within the same batch**"* / (line 279) PULS =
  **Sarathi-Serve 의 PIM 확장**(token budget 로 decode+chunked prefill 혼합).
- 검증 결과 풀 모델이 맞으면 §아래 설계로 진행. 아니면 사용자와 재논의.

## 1. 왜 persistent-mb 가 근본 문제였나 (이번 세션 결론)

- **현 코드**: admission 이 요청 묶음을 한 mb 로 만들고, 그 mb 가 *함께 prefill → 함께
  decode* 하는 생애를 강제(`_recompose_mb` 가 같은 요청 집합을 매 cycle 재구성).
  → 한 mb 안 요청이 **같은 생애 단계**라 초기엔 전부 prefill / 후기엔 전부 decode.
  지속적 혼합이 안 됨. 그걸 억지로 섞으려 `_try_join`(유휴율 게이트)을 붙임.
- **거기서 STEP 1~5.5 문제 전부 파생**: 단일 mb 캐파 독점 → per-mb KV 예산(KV/2),
  freed KV 를 한 mb 가 backfill → 합류 cannibalization, 유휴율 게이트 → TBT 폭증, …
  **전부 mb-컨테이너의 부작용.** 풀 모델이면 애초에 안 생긴다.

## 2. 올바른 모델 — 전역 풀 + 매 iteration 혼합 배치 (Sarathi 식)

- **전역 풀(running set)**: 요청들이 각자 단계로 존재 — *prefill 중*(남은 프롬프트 chunk
  필요) / *decode 중*(이미 prefill 끝, 매 step 1토큰). 요청별 생애(prefill→decode)는
  *요청의 속성*일 뿐, **배치 멤버십과 분리**된다.
- **매 iteration 배치 = 풀에서 선택**:
  - 살아있는 **decoder 전부**의 decode 토큰(→ PIM decode-attn) — TBT 위해 매 step 진행.
  - 거기에 **prefill chunk** 를 **GPU 시간이 PIM 시간과 같아질 만큼만**(슬랙에 숨겨) 추가.
  - prefill 끝난 요청은 **풀에 decode-only 로 복귀** → 다음 iteration 부터 PIM 채움.
- **F2 staggering** = 연속 iteration 배치를 2-μ-batch lookahead 로 overlap(PIM 이 M
  attention 하는 동안 GPU 가 M+1 QKV) — ARCH §5.6/§6.3. persistent 컨테이너 불필요,
  *연속 배치가 곧 M·M+1*.

## 3. 왜 이게 TBT·TTFT 를 동시에 잡나 (+ 유휴율=시간 동치)

- **TBT**: 모든 decoder 가 매 iteration 1토큰 → TBT = iteration 시간 ≈ t_pim
  (prefill 을 슬랙에 숨기니 cycle 안 늘어남). 바운드. (동시 decoder 수가 배치 한도
  넘으면 admission control 로 큐잉 — 표준 throughput/latency trade-off.)
- **TTFT**: prefill 이 decode 가 만든 GPU 슬랙에 숨어 진행 → TBT 안 깨고 prefill 진척.
- **유휴율 vs 시간**: 풀 모델 + **순간(per-iteration) 유휴**면 둘은 *동일*하다 —
  GPU 순간 유휴 = (t_pim − t_gpu)/t_pim → 이를 0 으로 = t_gpu→t_pim = 시간 균형.
  이전 세션이 "유휴율 나쁘다"고 본 건 **누적(cumulative) 유휴 + mb 파편화**의 산물.
  → 단, t_pim 은 dispatch 전 *계산*되므로(§6.3 computed wait) **시간 기준이 선제·정확**
  (per-iteration 유휴는 1틱 지연 반응형). 수렴점은 같으니 **시간 기준으로 구현**.

## 4. 이번 디버깅에서 *가져갈* 것 (carry-over — 재설계에도 유효)

- **단위**: `PIMExecutor.op_time` = **ns**, clock·GPU op_time = **µs**. dispatcher 가 PIM
  ×1e-3(`_op_time`), `_make_t_pim_fn` 도 ×1e-3(STEP 5 에서 수정, commit 1a1feb5).
  새 코드 시간 비교 시 단위 맞춰라.
- **op-time 물리(ctx 70K, diag_optime 로 직접 산출)**:
  - PIM decode-attn(27요청) ≈ **7.74µs**, GPU projection(QKV+O_PROJ) ≈ **6.18µs** →
    순수 decode 에선 PIM/GPU = ctx/56,160 = 1.25 (**PIM-bound**).
  - **PREFILL_ATTN = O(chunk×ctx)** 폭발적 — 3요청·chunk170·ctx70K ≈ **444µs**(PIM 의
    57배). **prefill 조금만 있어도 GPU 압도.** → prefill 은 *반드시 PIM 슬랙 안에*.
- **지표 = TBT·TTFT (idle 아님)**. idle_telemetry 는 측정·진단용으로만.
- **밸런스의 진짜 정의** = "유휴율 맞추기"가 아니라 **"prefill 을 PIM 시간 그늘에 숨겨
  cycle(TBT) 안 늘리고 throughput"**. 풀 모델에서 매 iteration prefill = max(0, t_pim −
  t_gpu)/per_token.
- **멤버십(어떤 요청, = 용량/큐) vs 사이클 일(prefill 얼마, = 시간) 분리** (배치_생애 §두 축).
- **PIM 가동률은 워크로드·물리 함수**(ctx/56K) — 저ctx GPU-bound·PIM 유휴는 정상.
  PIM 가치는 가동률 아니라 op-level(Aux2 버스절감·F5). 타깃 = **long-context agentic**.
- **decode 는 충전 불가(부산물)** — 풀에서도 decode 일감은 "이미 prefill 끝난 요청"이
  공급. 밸런스 레버는 **prefill 양**뿐(§D-3 논리 유효).
- **스케일 스펙트럼 실측**(REPORT §14): T-S/T-GEN(저ctx) GPU-bound·PIM 유휴 / agentic
  OFF(고ctx 순수 decode) PIM 활용(유휴 46%·둘 다 ~60%) / agentic ON 합류가 GPU 과포화.

## 5. 무엇이 *moot* 되나 (persistent-mb 밴드에이드 — 풀 모델이면 불필요)

> 코드 *substrate* (op-time 산식, dispatcher, PIMExecutor, KVAccountant, Completion, DAG
> 노드, IdleTelemetry, InstancePipeline)는 재사용. **스케줄링/구성 레이어만 교체.**

- `_per_mb_kv_budget`·`_STAGGERING_TARGET_MB`(per-mb KV 예산) — 풀이면 KV 는 전역 풀
  한계라 per-mb 분할 불필요.
- `_try_join` + 유휴율 게이트 + hysteresis(idle_theta_low/high) — 풀이면 멤버십=용량,
  밸런스=시간이라 합류·게이트 개념 자체가 없어짐(매 iteration 풀에서 재선택).
- `balance_intra_A`(유휴율 chunk 증량) — 시간 기준 prefill 사이징으로 대체.
- persistent `MicroBatch` 컨테이너·`_recompose_mb`·window 의 mb 등록/evict — per-iteration
  배치 형성으로 대체. (단위 수정 1a1feb5 는 유효, 나머지 STEP 1~5.5 의 mb 관련 변경은
  상당수 대체됨 — REPORT 가 그 여정을 기록.)

## 6. 할 일 (검증 → 설계 → 구현 → 측정)

1. **검증** — ARCH §6 정독, 풀 모델 해석 확정(§0). 아니면 사용자와 재논의.
2. **설계** — running pool(prefill-queue + decode-set) + per-iteration 배치 형성기:
   - decoder 전부 선택(또는 배치 한도까지) + prefill = PIM 슬랙(시간 기준, base floor 없음).
   - F2 = 2-μ-batch lookahead(연속 배치 overlap). admission control(동시 decoder 한도).
   - prefill→decode 전이 시 풀 복귀. TTFT↔TBT 정책(슬랙 0 시 최소 prefill 보장 여부).
3. **구현** — 스케줄링 레이어 교체(substrate 재사용). 단위테스트 + 회귀.
4. **측정** — measure_steady 에 **TBT·TTFT 산출 추가**(decode 토큰당 cycle 시간 / 첫
   토큰까지 시간). 스케일 스펙트럼(T-S/T-GEN/agentic)으로 풀 모델의 TBT·TTFT 확인.
   - **트레이스 포맷은 그대로**(`arrived_at, prefill, decode` = 실제 요청; 한 요청 안
     prefill→decode 종속은 진짜). 재생성 불필요. 풀 모델은 *스케줄러*가 풀에서 매
     iteration 선택하는 것이지 워크로드 표현이 바뀌는 게 아님.
   - **단 도착 패턴 재고** — 현 트레이스는 버스트 도착(상대적으로 전부 동시)이라 풀에서도
     "다 같이 prefill → 다 같이 decode" 전이만 보임. **풀 모델의 지속적 혼합 정상상태**를
     보려면 도착을 *처리 시간축에 맞춰 흩뿌린(staggered)* 트레이스를 추가 — 일부 decode 중에
     새 요청이 도착해 prefill → 항상 굴러가는 mix. 포맷 아니라 **도착 분포(arrival span)만**
     길게 재생성. (필수 아님 — 버스트로도 동작하나 전이 구간만 관측됨.)
5. **문서** — 배치_생애·README·REPORT 를 풀 모델로 갱신. README "Target Workload" =
   long-context agentic 재프레이밍.

## 7. 코드 지도 (현재)

- `src/puls_sched/` — `admission.py`(layer1·balance_*), `main_loop.py`(mb 관리·
  _recompose_mb·_try_join·_make_t_pim_fn·_compose_admission_payload), `dispatcher.py`
  (_op_time: GPU 초×1e6, PIM ns×1e-3·cycle 구조), `pim_emulator.py`(op_time=ns),
  `micro_batch.py`, `window.py`, `config.py`(AdmissionConfig), `idle_telemetry.py`,
  `kv_accountant.py`, `completion.py`, `dag.py`, `forward_pass.py`, `run.py`.
- `debug_phase1/` — `measure_steady.py`(엄밀판 측정), `diag_optime.py`(op-time 직접
  산출 — 출력 단위 ns/µs 주의), `gen_agentic.py`/`gen_general.py`/`gen_step5_traces.py`
  (트레이스), REPORT_baseline.md §12~14, 배치_생애.md(repo 최상위).

## 8. 현재 상태 (커밋 7개, 미푸시 — 로컬)

```
113c4e2 docs: STEP6 §D-3 (구버전 프롬프트, 이 파일로 대체됨)
04807bd docs: 두 축(멤버십=용량 / 밸런스=시간 / 유휴율=결과)
e1e0cd2 docs: (구) STEP 6 핸드오프
3bcbaaa docs: STEP 5 스케일 스펙트럼 + 발견
1a1feb5 fix: ns/µs 단위 (balance t_pim_fn)   ← 유효(carry-over)
b9f3ffc feat: per-mb KV 예산                  ← 풀 모델이면 대체
9368edf feat: hysteresis deadband             ← 풀 모델이면 대체
```
working tree 깨끗(untracked `analysis/`·`_scratch_*` 무관). 테스트: 풀 회귀는 재설계
커밋 게이트에서. (배치_생애 §두 축, REPORT §12~14 가 결론 기록.)

## 9. 작업 습관 당부 (이번 세션 실수 — 꼭 지켜라)

- **모든 bash 에 `cd /c/Users/rhs02/Desktop/puls-rfc/implementation &&` 붙여라**
  (cwd 불안정 — 5번쯤 빠뜨려 크래시).
- **`run_in_background` 안에서 `&` 금지**(detach → 알림 깨짐).
- **추측 말고 측정/코드로 확인**(이번 세션 가설 5번 빗나감; op-time 은 `diag_optime`로,
  단위 ns/µs 조심).
- **한 메시지에 도구 과병렬 금지**(cascade 취소). 무거운 측정은 백그라운드 하나씩.
- **변경하면 바로 커밋.** 회귀는 개발 중 가벼운 타깃만, 커밋 직전 풀 1회.
- **PYTHONIOENCODING=utf-8 + 파일 출력**(콘솔 cp949 깨짐). 커밋 메시지는 bash heredoc.
- **재설계는 큰 변경 — substrate 재사용, 스케줄링 레이어만 교체.** ARCH §6 정독부터.

시작 전 위 문서들 다 읽고(특히 **ARCH §6 검증**), **(0) 풀 모델 해석 확정 → (1) 설계 →
(2) 구현 → (3) TBT·TTFT 측정**의 접근 계획을 먼저 제시해라.
