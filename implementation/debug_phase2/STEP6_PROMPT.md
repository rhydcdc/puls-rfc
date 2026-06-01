# STEP 6 — Phase-2 풀 모델 구현 (새 대화용 프롬프트)

> 아래를 새 대화창에 그대로 붙여넣으세요. 설계는 끝났고 **구현(S2 마무리 → S5)**만 남았습니다.
> 단일 기준 문서 = [`debug_phase2/PLAN.md`](PLAN.md) (살아있는 체크리스트). 측정 메모 =
> [`debug_phase2/REPORT.md`](REPORT.md).

---

PULS 스케줄러 Phase-2(풀 모델 재설계)의 **구현**을 이어서 한다. 설계·핵심 물리는 이미
확정됐다(이전 세션). **시작 전 `implementation/debug_phase2/PLAN.md` 를 정독**하고, 거기
체크리스트(`- [ ]`/`- [x]`/`- [~]`)대로 진행하라. 작업하며 PLAN.md 를 계속 갱신한다.

## 0. 무엇이 이미 끝났나 (재작업 금지 — 그대로 위에 쌓는다)

커밋 11개 로컬(미푸시). 코드 커밋 3개는 **확정·유효**:
- **S0** (`7c00532`) — Instance B FFN 을 스케줄 노드로 모델링. `NodeType.FFN`(node.py),
  `O_PROJ→FFN` edge(dag.py), `INSTANCE_B` 자원 + `pick_instance_b`/`dispatch_instance_b` +
  I6(dispatcher.py), layer advance 트리거 O_PROJ→**FFN**(main_loop.py), `check_I6`. 테스트
  `tests/test_phase2_ffn_stage.py` 9개 green. → **F3(인스턴스 A∥B) 동역학 발현.**
- **S1** (`fe7b7ac`) — `admission.balance_intra_A`(유휴율 레버) 삭제. `_try_join`→
  `_backfill_slot`(유휴 게이트 없는 연속 backfill), `_join_gate_open` 삭제.
- **TP=8 픽스** (`40e812a`) — ★ **가장 중요.** `compute_gpu_op_time_s`·`compute_ffn_op_time_s`
  가 단일 GPU peak 으로만 나눠 GEMM 을 8배 과대평가하던 버그 수정(`num_gpus` 파라미터,
  A=8·B=8). PIM 은 k_aggregate(8-GPU 채널 합산)로 이미 분산 반영 → 이제 PIM·GPU·B 가
  **같은 8-GPU 기준·같은 µs 단위**로 통일. (검증 완료: GPU QKV 3.43µs→0.43µs 정확히 ÷8.)

> **단위 규약 (반드시 지켜라):** PIM `op_time()` 반환 = **ns** → ×1e-3 = µs. GPU/FFN
> `compute_*_op_time_s` 반환 = **초** → ×1e6 = µs. clock = µs. GPU/FFN 은 `num_gpus=8` 전달
> 필수(안 하면 8배 과대). 새 시간 비교마다 단위·÷8 확인.

## 1. 확정된 동작점 (PLAN §0.8 — 이게 구현의 목표)

**스케줄러의 레버 = "배치에 넣는 디코더들의 KV 길이 합" = PIM KV 총량.** 디코드는 넣기는
자유지만 **쪼개 넣기 불가**(요청 KV 통째) → 정확히 한 점에 못 맞추니 **오차 범위**로 수렴.

| 파라미터 | 값 |
|---|---|
| **목표 KV 총량** | **25M 토큰** → PIM≈GPU-A≈B≈101µs (spread 0.6%) |
| **허용 범위(15%)** | **21.5M ~ 29M 토큰** |
| **prefill** | **512 토큰/배치 고정** (2^9). 동적 사이징 없음 (1024+ 는 GPU-A 폭주로 균형 불가, REPORT) |
| **KV 캐파** | **30M 토큰** (hard ceiling, 넉넉) |
| 균형 시간 X | ~101µs / N_dec = 부산물(평균 ctx 100K면 ~248개) |
| 타깃 워크로드 | long-context agentic, 요청 ctx ~87K~117K |

- 세 시간 함수: `t_PIM`=f(Σ디코더 KV, 개별 ctx 무관·합만) / `t_GPU-A`=proj(batch)+
  PREFILL_ATTN(prefill×ctx) / `t_B`=FFN(batch=N_dec+prefill).
- 저ctx(<7K)·짧은 풀: KV 합이 25M 못 미쳐 작은 배치 → PIM 노는 게 **정상**(물리, 고칠
  대상 아님). length-aware 별도 정책 불필요(동작점이 흡수, §0.9).

## 2. 핵심 통찰 — 동작점 고정이 코드를 **단순화**한다 (PLAN §2.5)

밸런스가 정적 동작점(KV 합 + prefill 512)으로 확정 → **매 tick 동적 측정·계산 기계장치가
통째로 moot.** former 가 "cycle 측정→비교→chunk 사이징"이 아니라 "KV 합 채우기 + prefill
512"로 단순화. main_loop ~100 LOC↓, admission 절반↓ 예상.

## 3. 남은 구현 (PLAN §3 의 S2~S5 — 이걸 한다)

### S2 — `main_loop.py` 동작점 former + 동적 밸런스 기계장치 제거 (먼저)
- **삭제(§2.5 moot):** `_compose_admission_payload`·`_measure_cycles`·`_make_t_pim_fn`·
  `_prev_a/b_active_snapshot` (cycle 측정). `admission.balance_inter_AB`·`balance_pim_slack`
  + `deadband.py` (동적 prefill 사이징).
- **`admission.layer1` 재작성:** t_proj·t_pim_fn·a_cycle·b_cycle·per_token·balance_* 인자/
  호출 전부 제거. 새 종료 조건 = **admit 한 디코더 Σkv ∈ [21.5M, 29M] 도달** + **prefill
  512 고정**. head-of-line walk + KV admit(can_admit) 골격은 유지.
- **ADMISSION_TICK 핸들러:** "spec→영속 mb 1개 생성"을 per-iteration 동작점 former 로 교체.
  payload trivial 화.
- **★ 반드시 유지(삭제 아님):** 생애 전이 `prefill_processed ≥ len(prompt)` 시
  `RequestState.PREFILL→DECODE` (main_loop:345·353). prefill 끝난 요청을 decode-only 로
  돌려 PIM 채움 — 풀 모델의 본질. "밸런스 계산은 정적으로 대체, 요청 생애는 그대로."
- **tick:** 주기 polling 은 이미 없음(Phase-1 STEP 2.5). 빈 슬롯 재충전 트리거(완료
  이벤트→former 재호출)는 유지 필수.

### S3 — window/F2 정합 + 옛 테스트 정리
- former 가 활성 슬롯 2개를 명시적으로 채워 F3·더블버퍼링 발현(capacity=3 유지). disjoint
  분할 추적(요청→슬롯). + 옛 telemetry `instance_pipeline.dispatch` 잔여 정리.
- **사전-깨짐 3개**(PLAN "알려진 사전-깨짐" 절): `test_instance_pipeline_dispatch_invoked_
  per_layer`(S0 가 hot path 제거 → 호출 0, FFN 노드 기준 재작성/폐기) + `test_cross_module_
  pipeline` 2개(PIM op_time ns 267.5 기대인데 µs 0.2675 반환 — 단위 기대값 수정).

### S4 — 측정 substrate (TBT·TTFT)
- `Request.first_token_time` 추가. 완료 요청 sink(현재 `in_flight_requests.pop` 후 버려짐,
  main_loop:358). L 도달 첫 decode 시 first_token_time 기록.
- `debug_phase2/measure_steady.py` 작성(Phase-1 것은 스텁). TTFT=`first_token−arrival`,
  TBT=`(completion−first_token)/(decoded−1)`. p50/p90/max. warm-start seed(§2.6) 플래그.

### S5 — 데드코드 정리 + 문서
- S2 orphan(import·`micro_batch.prefill_chunk_budget` 등) 제거. KV 캐파 4M→30M(config.py
  `default_dummy_config`).
- `배치_생애.md`·`README.md`(Target Workload=long-ctx agentic) 갱신. `debug_phase2/REPORT.md`
  에 측정 결과.

## 4. 측정 성공 기준 (PLAN §5 — idle ↔ spread)
완벽 overlap 이면 cycle=max(PIM,GPU-A,B), 자원 idle=(max−busy)/max → **spread = 가장
한가한 자원의 idle**. 검증:
- **완전 균형(KV 25M)**: 세 자원 idle ≈ 0%. 의도한 최적점.
- **허용 범위(21.5~29M)**: 가장 한가한 자원 idle ≤ 15%.
- **이론↔실측 일치 = Phase-2 핵심 성공 판정** (실측 idle 이 이론 ≈0~15% 에 가까운지 =
  F2/F3 발현 증거). 스케일 스펙트럼(저/중/고ctx)으로 확인.

## 5. 작업 습관 (이전 세션 실수 — 꼭 지켜라)
- **도구 과병렬 금지** (cascade 취소 발생함). 무거운 측정은 백그라운드 하나씩.
- **검증은 철저하되 과도하지 말 것**: 변경 모듈+직접 영향 테스트만 철저히, 무관 전체
  스위트 반복·과잉 케이스 금지(PLAN §7).
- **추측 말고 코드/측정으로 확인** (가설 여러 번 빗나감). op-time 은 직접 산출.
- **변경 즉시 커밋**(heredoc 메시지), `PYTHONIOENCODING=utf-8` + 파일 출력(콘솔 cp949 깨짐).
- 옛 전체 회귀 안 돎(다 교체 중) — 변경별 타깃 테스트만. 깨진 옛 테스트는 풀 모델에 맞게
  갱신(이유 주석).
- bash cwd 불안정 — 한글 디렉터리 나열은 Glob/Read 로.

## 6. 코드 지도 (현재)
- `src/puls_sched/` — `main_loop.py`(510L, ADMISSION_TICK·former·생애전이·삭제대상 측정기계),
  `admission.py`(159L, layer1·balance_pim_slack[삭제]), `dispatcher.py`(277L, FFN 노드·
  INSTANCE_B·`_op_time` num_gpus), `config.py`(compute_gpu/ffn_op_time_s num_gpus·
  default_dummy_config[KV cap 4M→30M]), `node.py`(NodeType 5: +FFN), `dag.py`(O_PROJ→FFN),
  `invariants.py`(I1~I6), `pim_emulator.py`(op_time=ns), `evaluator.py`(f3 num_gpus·측정),
  `idle_telemetry.py`, `kv_accountant.py`, `completion.py`, `request.py`(FSM), `run.py`.
- `debug_phase2/` — `PLAN.md`(기준), `REPORT.md`(prefill sweep), `STEP6_PROMPT.md`(이 파일).
- `tests/` — `test_phase2_ffn_stage.py`(S0, 9 green) + 옛 테스트 다수(S3 정리).

시작: PLAN.md 정독 → S2(former + 측정기계 삭제)부터. 변경마다 타깃 테스트 + 즉시 커밋.
