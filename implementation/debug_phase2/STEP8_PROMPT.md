# STEP 8 — Phase-2 풀 모델 확정 후 S5 정리 (새 대화용 프롬프트)

> 새 대화창에 그대로 붙여넣으세요. **알고리즘·측정 끝. 남은 건 S5 정리(문서·이동·폴더분리·데드코드).**
> 단일 기준 = [`OPERATING_POINT.md`](../../OPERATING_POINT.md) (canonical spec).
> ⚠ `STEP6_PROMPT.md`·`STEP7_PROMPT.md` 는 이전 핸드오프 — **superseded, 볼 필요 없음. 이 STEP8 만.**

---

PULS 스케줄러 Phase-2(풀 모델)의 **구현·검증은 끝났다.** 남은 건 S5(정리·문서). **시작 전 정독:**
1. **[`OPERATING_POINT.md`](../../OPERATING_POINT.md)** — 동작점·배치 구성 알고리즘 *정답지*(canonical). 정독.
2. **현 풀 모델 코드** (§2) — `src/puls_sched/admission.py`(steer_decode_set), `main_loop.py`
   (_refill_pool·_compose_microbatch·_recompose_mb·_populate_mb_phases), `debug_phase2/measure_steady.py`.
3. **[`PLAN.md`](PLAN.md)** 진행 로그 (맨 아래).

> ⚠ **`배치_생애.md` 와 `ARCHITECTURE.md` §6 은 옛 모델(S2 sticky-cohort / §6.4 유휴율 adaptive)을
> 서술 — *폐기됨*. 현 코드의 기준 아님(= S5 갱신 대상). 절대 그걸 현행으로 읽지 말 것.**

## 0. 이미 끝난 것 (재작업 금지)

핵심 결론 = **풀 모델로 동작점(idle≈0) 검증 완료.** 커밋(로컬, 미푸시):
- `54ee4d6` **★ 풀 모델 rework** — former-v2 를 OPERATING_POINT §3 풀 모델로. (이전 S2 sticky-cohort 폐기.)
- `8255886` **S4(d) 측정** — measure_steady 풀모델판 + gen_sweep_traces. **idle spread 4.7% 검증.**
- `58a9e79` **README** 한/영 갱신 (디버깅중 제거·워크로드 무가정·Runtime Validation).
- (이전) `bcd8492` prompt_len 경량화 + prefill liveness, `9df1bc7` 사전-깨짐 8삭제+5마이그,
  `8b74e56` S4(b) first_token_time+완료 sink, `9118624` S4(c) measure_steady 초판.
  ※ `1acb8b6`(decode steering)·`898177b`(prefill steering)의 `layer1` 은 `54ee4d6` 풀 모델이 대체.

**검증 수치 (커밋 8255886):** 대량-상주 decode 풀(긴 decode) 워크로드서 한 μ-batch =
decode **122/123** · Σkv **12.33M**/12.3M · prefill **256** · depth-work 27.1M / 자원 idle
GPU-A 8.0% · PIM 12.6% · FFN 12.6% · **spread 4.7%** (converged). 직렬 ~67% 대비 1/14 →
F2(double-buffering)·F3(inter-instance pipeline) 발현 = idle≈0. 드레인 정확성: 합성 acceptance
(c1·c3·c4·run) green(전부완료·KV누수0·종료). 직전 full suite 850 green(실트레이스 e2e 3건 제외).

## 1. ★ 현 풀 모델 (S5 문서 갱신의 기준 — 정확히 이거다)

OPERATING_POINT §3 그대로. **세 관심사 분리** (이전 S2 는 셋을 한 cohort 로 뭉쳐 틀렸음):

1. **Admission = 풀 보충만** (`main_loop._refill_pool`) — request_queue → in_flight(PREFILL),
   KV 게이트(can_admit)만. 디코드/prefill 타깃 *무관*. prompt_len=0(decode-only)·이미 prefill
   끝난 요청은 즉시 DECODE 전이.
2. **decode-set 구성** (`admission.steer_decode_set(candidates)`) — 인플라이트 **DECODE 풀**서
   로컬 그리디 steering+age-cap 으로 (개수 123, Σkv 12.3M) 선택. **순수 선택**(KV admit·큐 조작
   없음 — KV 는 풀 진입 시). 미선택 wait++, 선택 wait=0.
3. **prefill 구성** (`main_loop._populate_mb_phases(prefill_pool, 256)`) — 인플라이트 **PREFILL
   풀**서 256 토큰을 depth-합 25.6M 되게 per-token greedy + age-cap 분배. **0토큰 요청은
   풀 잔류**(prefill_chunk 에 안 넣음 — 빈-chunk 멤버십 유지는 *제거됨*; 넣으면 mb 부풀려
   prefill 과분산·decode 고갈). 별개 축.
4. **μ-batch 생애** — `_compose_microbatch`(새 mb) / `_recompose_mb`(forward pass 후 풀에서
   재선택). disjoint(`_assigned_ids` 로 다른 mb 멤버 제외). window 활성 목표 `_STAGGERING_TARGET_MB`
   =2 (`_fill_window`; capacity 3 = 2 active + 1 전이 여유). = S2 가 지운 per-iteration 재선택 복원.

**핵심 통찰 (S5 문서에 반영):**
- **워크로드 무가정** — 길이분산 무관 steering. decode·prefill 이 풀에 *풍부* 한 어떤 분산-서버
  워크로드든 동작점 수렴. **"타깃 워크로드"·"평균 100K" 같은 건 없다**(100K=12.3M/123, KV 캡 유도용).
- **상주 decode 풀이 관건** — 실서버엔 기존 디코더 대량 상주(긴 생성·고동시성). idle≈0 은 풀이
  풍부할 때의 균형. cold-start throughput(prefill 이 디코더 생성) 지속성은 *별개 축*(긴 decode
  아니면 풀 못 채움) — warm-start seed 가 *상주 풀* 을 대표(측정 substrate).

## 2. 코드 지도 (풀 모델 반영)

- `src/puls_sched/admission.py` — `steer_decode_set`(decode-set 순수 선택). `MicroBatchSpec` 은
  이제 **사문**(main_loop 미사용; test_admission/test_meta field-check 만 참조 → S5 정리 검토).
- `src/puls_sched/main_loop.py` — `_refill_pool`·`_assigned_ids`·`_select_decoders`·`_prefill_pool`·
  `_build_mb_fields`·`_compose_microbatch`·`_fill_window`·`_recompose_mb`·`_populate_mb_phases`
  (prefill 분배)·`_maybe_advance_forward_pass`(FFN 완료→토큰 생성·완료 sink·재구성·fill_window).
  `_fire_admission_tick`(진단). `completed_requests` sink(S4b).
- `src/puls_sched/request.py` — FSM. `prompt_len`(int, 경량화), `wait`(decode age-cap),
  `prefill_wait`(prefill age-cap), `first_token_time`(S4b).
- `src/puls_sched/config.py` — `decode_count_target=123`·`kv_operating_target_tokens=12.3M`·
  `prefill_chunk_default=256`·`prefill_kv_work_target_tokens=25.6M`·`age_cap=2`·
  `kv_capacity_aggregate=30M`. **사문 가능 필드(S5 검토):** `n_sat`·`ctx_tier_*`·`deadband_width`·
  `idle_theta_*`·`pim_slack_safety_margin`·`gpu_op_time_per_token_us`.
- `debug_phase2/measure_steady.py` — TTFT/TBT/idle + 대표 warm-start seed(사이클-시간 가중) +
  수렴-정지 + 배치 구성 진단. `gen_sweep_traces.py` — 스케일 스펙트럼 트레이스 A/B/C/D 생성
  (data/sweep_*.csv 는 재생성 가능, 미커밋).
- `dispatcher.py`(FFN·INSTANCE_B·num_gpus)·`pim_emulator.py`(op_time=ns)·`node.py`(NodeType 5)·
  `dag.py`(O_PROJ→FFN)·`window.py`(capacity 3)·`completion.py`(finalize=KV release).
- **vestigial(S5 데드코드 후보):** `deadband.py`, `instance_pipeline.py`+`forward_pass.ForwardPass`
  (hot path 미사용 — SchedulerCore 는 LayerState.advance 만), `idle_telemetry` 의 deadband 잔재.

## 3. 남은 작업 (S5 정리)

> 원칙: surgical, 변경마다 타깃 테스트 + 즉시 커밋. **풀 스위트 반복 금지**(직전 정합이면 타깃만).

- [~] **문서 갱신 (풀 모델 기준):**
  - [x] `ARCHITECTURE.md` §6(스케줄러) — 옛 §6.4 유휴율 adaptive·deadband → **풀 모델**로 교체
    (admission 풀보충 ∥ decode-set ∥ prefill, per-iteration 재구성, window 2-active) **완료**
    (`4675ef0` 영문 / `3bb0136` 한글). §5.2 ragged 정정·§6.6 intro·TOC 동반. F1~F5·인스턴스
    분리·DAG 골격 유효 유지. **§6.8 idle floor 증명 상세 신설**(아래 ★ 항목 결과). 정합 문단
    추가(풀 모델 = 실 continuous-batching 패러다임, sticky-cohort 가 비정합 — 사용자 확인).
  - [ ] `배치_생애.md` — **삭제**(사용자 확정 2026-06-02: OPERATING_POINT 가 근본 문서가 되므로
    옛 모델 서술 문서 불필요). 코드 주석 2곳(main_loop.py:54 §밸런스, window.py:17 §세 한계)을
    OPERATING_POINT 참조로 repoint 후 삭제. **(미완 — 남음)**
  - [x] `README` (한/영) — 풀 모델 + floor 결과 간단 반영 완료(`58a9e79` 풀모델 + `4675ef0`/`3bb0136` floor).
- [x] **OPERATING_POINT.md → repo 루트로 이동** — `implementation/debug_phase2/` → repo 루트
    (README·ARCHITECTURE 층위). 양방향 링크 갱신 완료(들어오는 7: REPORT·PLAN·STEP7/8 →
    `../../OPERATING_POINT.md`; 나가는 2: proto_steering → `implementation/debug_phase2/`). 코드
    주석의 텍스트 멘션("OPERATING_POINT §X")은 경로 아님 → 무변경. canonical 격상 완료.
- [skip] **src 폴더 분리** — 건너뛰기(사용자 확정 2026-06-02: ~15 모듈 flat 이 단순, 분리 이득 작음).
- [~] **데드코드 정리** — 안전분 완료, 위험분 연기(사용자 확정 2026-06-02):
  - [x] `MicroBatchSpec` 제거(`d877881`) — 사문 클래스, hot path 미사용.
  - [x] config 고립 필드 3개 제거(`fe77fae`) — `idle_theta_low/high`·`pim_slack_safety_margin`·
    `gpu_op_time_per_token_us`. test_meta `_EXPECTED_ADMISSION_FIELDS` 동반 갱신.
  - [!] `deadband.py` — **데드 아님(유지)**: evaluator(라이브, report 산출)가 `in_band`/`lookup_width`
    를 수렴 진단(`in_band_fraction`)에 사용. §6.4 폐기분은 *admission 제어용* deadband, evaluator
    진단분은 별개로 live. 지우면 동작(report) 변경 → 데드코드 아님.
  - [ ] `instance_pipeline`/`forward_pass.ForwardPass` — **진짜 dead-wiring 확인**(라이브는
    LayerState.advance + dispatcher.dispatch_instance_b; ForwardPass.run 미사용). **연기** — run.py
    재배선 + ~10 테스트 파일 정리 + 풀 스위트 1회 검증 필요한 refactor. 거동·floor 영향 0(잔존).
    별도 세션. (LayerState 는 forward_pass.py 에 동거 — 모듈 통삭 불가, ForwardPass 클래스만 대상.)
  - config `n_sat`·`ctx_tier_*` — 보류(n_sat=main_loop 주석·test_aux1 참조; ctx_tier=deadband wiring).
- [x] **REPORT.md** — floor 증명 + prefill sweep 정정("512만 균형" 폐기 → family, OPERATING_POINT
    §5.2) **완료**(`d1e48a3`). (throughput/상주풀 통찰은 README Runtime Validation 에 이미 있음.)
- [x] **실트레이스 시뮬레이션 테스트 skip** — 완료(`4750e48`). 사용자 확정(안 쓰되 삭제 X). 실 trace
    full cold-start 시뮬 4건 `@pytest.mark.skip`(prefill 47K~2.5M = 수억 step 비현실): e2e
    TestE2eRealTrace(4), lifecycle real_longbench_100·capacity_bumped_500, stress_real_500. 빠른
    로더/accounting/stats 단위테스트는 유지(sweep 파서 검증에도 유효). 검증 = sweep_* + warm-start.
- [x] **★ idle floor 증명** — **완료**(`d1e48a3`, `analysis/floor_proof.py` + REPORT §1–5). 4 sub-item
    전부:
  1. [x] **이론 floor** — live μ-batch 에 dispatcher 와 동일 op-time 함수 직접 호출(합성 오차 0).
     t_gpuA 52.89(병목) / t_pim 50.43 / t_ffn 50.45 → cycle=max → floor: GPU-A 0% · PIM 4.65% ·
     FFN 4.61%, **floor spread 4.65%**.
  2. [x] **측정−이론 = overlap gap** — 측정 spread 4.62% ≈ 이론 4.65%; overlap gap 세 자원 **균일
     8.0%**(병목 GPU-A 측정 idle = fill/drain·staggering). 미설명 잔여 0.
  3. [x] **prefill 선택 최적성** — depth-work 27.1M(+5.8%)가 t_gpuA 의 80%(PREFILL_ATTN). 풀 얕은
     후보 ~1.3 존재. counterfactual: 25.6M 명중 시 floor spread 0.26%.
  4. [x] **age-cap 분리** — `age_cap=∞` 면 steering 이 depth-work 25.73M 명중 → spread **0.11%**(단
     starvation). ⇒ floor spread 4.6% = **age_cap=2 의 공정성 비용**(풀 고갈 아님), 의도된 trade-off.
  → REPORT 기록 완료. **"알고리즘 floor 도달" 확정**(측정 = 이론 floor + 설명된 overlap gap).
- [ ] **★ 새 시드 풀 idle 관찰 (cleanup 후 *마지막* 실행)** — `floor_proof.py --seed <new>` 로 warm-start
    풀을 새 난수로 다시 뽑아 idle 재측정. 새 풀에서 idle 이 *증가*(다른 불균형)하는지, 혹은 *우연히
    age-cap 미발동으로 하락*하는지 관찰 → floor 결과의 견고성(seed 비의존) 확인. 모든 정리·이동·
    데드코드 제거가 끝난 *뒤* 실행해 회귀까지 겸사 검증.
- [ ] 자기리뷰(커밋 전): 옛 심볼 grep 0 / `python -c "import puls_sched"` 순환 0 / 모듈 LOC /
    플랜 인벤토리 vs diff / 미사용 import·죽은 분기 0.

## 4. 작업 습관 (이전 세션 실수 — 꼭 지켜라)
- **추측 말고 코드/측정으로 확인.** ★ 심볼 삭제 시 tests *및 src 호출처* 모두 grep.
- **도구 과병렬 금지**(cascade 취소). **풀 스위트 반복 금지** — 변경별 타깃 테스트만.
- **변경 즉시 커밋**(heredoc, `Co-Authored-By`), `PYTHONIOENCODING=utf-8` + 파일 출력(콘솔 cp949 깨짐).
- bash cwd 불안정 — `cd /c/Users/rhs02/Desktop/puls-rfc/implementation &&` 로 시작. 한글 경로는 Glob/Read.
- **단위 ns/µs** — PIM op_time=ns(×1e-3=µs), GPU/FFN num_gpus=8 전달 필수. 새 시간 비교마다 확인.
- **★ OPERATING_POINT 가 단일 기준** — ARCH/배치_생애/§0.8 과 충돌 시 OPERATING_POINT 가 이김.
  "타깃 워크로드"·"balanced 100K" 표현 금지(밸런스=타깃 4개 명중). 깨진 옛 테스트는 풀 모델에
  맞게 갱신(이유 주석), 삭제된 기능 테스트는 폐기.

시작: OPERATING_POINT.md + 풀 모델 코드(admission·main_loop·measure_steady) 정독 → S5 중 원하는 것부터.
문서 갱신은 §1 풀 모델 요약을 기준으로. 변경마다 타깃 테스트 + 즉시 커밋.
