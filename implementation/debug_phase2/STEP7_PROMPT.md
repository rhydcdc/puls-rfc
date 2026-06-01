# STEP 7 — Phase-2 former-v2(steering) 구현 + 측정 (새 대화용 프롬프트)

> 새 대화창에 그대로 붙여넣으세요. **설계·검증 끝. 구현(former-v2 → S4 → S5)만 남음.**
> 단일 기준 = [`OPERATING_POINT.md`](../../OPERATING_POINT.md) (배치 구성 알고리즘 canonical spec)
> + [`PLAN.md`](PLAN.md) (살아있는 체크리스트). 측정 메모 = [`REPORT.md`](REPORT.md).
> ⚠ `STEP6_PROMPT.md` 는 이전(FIFO+skip 시절) 핸드오프 — **superseded, 볼 필요 없음. 이 STEP7 만 보면 됨.**

---

PULS 스케줄러 Phase-2(풀 모델)의 **구현**을 이어서 한다. 설계·핵심 물리·배치 구성 알고리즘은
모두 확정·검증됐다(이전 세션, 프로토타입). **시작 전 반드시 정독:**
1. **[`ARCHITECTURE.md`](../../ARCHITECTURE.md)** — 전체 의도(§3.4 인스턴스 A∥B, §5.6 더블버퍼링,
   §6 스케줄러). ⚠ **ARCH §6.4 는 *유휴율(idle-fraction) 기반 adaptive admission* 을 서술하지만,
   이후 설계가 *정적 동작점 + steering* 으로 진화했다.** 밸런스 로직이 ARCH 와 약간 다르니
   **구현은 OPERATING_POINT.md 대로 하라**(ARCH 의 idle-fraction 기계장치는 S2 에서 이미 삭제됨,
   유휴율은 진단 출력으로만). ARCH 의 F1~F5·인스턴스 분리·DAG 골격은 그대로 유효.
2. **[`OPERATING_POINT.md`](../../OPERATING_POINT.md)** — 동작점·배치 구성 알고리즘의 *정답지*. 정독.
3. **[`PLAN.md`](PLAN.md)** — 진행 로그(맨 아래) + 체크리스트.

## 0. 이미 끝난 것 (재작업 금지)

코드 커밋(로컬, 미푸시):
- **S0** (`7c00532`) — Instance B FFN 을 스케줄 노드로(`NodeType.FFN`, `O_PROJ→FFN` edge,
  `INSTANCE_B` 자원, I6). layer advance 트리거 O_PROJ→**FFN**. F3 동역학 발현.
- **S1** (`fe7b7ac`) — `admission.balance_intra_A` 삭제.
- **TP=8 픽스** (`40e812a`) — GPU/FFN op-time 을 8-GPU 분산(`num_gpus=8`)으로. PIM 은 2048채널
  aggregate. **세 자원 동일 8-GPU·µs 기준 통일.**
- **S2** (`5600955`) — 동작점 former + 측정/밸런스/backfill/max_batch/mfu 제거. `admission.layer1`
  = "Σkv ≥ 목표까지 admit"(아직 *개수 통제 없음* — former-v2 가 보강). config `kv_capacity_aggregate`
  4M→60M, `kv_operating_target_tokens` 추가. backfill·deadband·balance 3종 삭제.
- **S3** (`7ec8dad`) — 사전-깨짐 테스트 O_PROJ→FFN 마이그레이션 + **run.py 회귀 수정**(S2 가 지운
  `_compose_admission_payload` 를 run.py:164 가 호출하던 버그). acceptance·e2e·lifecycle green.

설계·검증(프로토타입, 코드 아직 미반영):
- 동작점·steering·age-cap 전부 op-time 직접 산출 + 프로토타입으로 검증 → OPERATING_POINT.md.
- `sweep_operating_point.py`(동작점 도출), `proto_steering.py`·`proto_steering_fair.py`(알고리즘 검증).

> **단위 규약(필수):** PIM `op_time()`=**ns**(×1e-3=µs). GPU/FFN `compute_*_op_time_s`=**초**(×1e6=µs).
> GPU/FFN 은 `num_gpus=8` 전달 필수(안 하면 8배 과대). 새 시간 비교마다 단위·÷8 확인.

## 1. 확정 파라미터 (OPERATING_POINT.md §1 — 이게 구현 목표)

| 파라미터 | 값 (prefill **256** 기본) |
|---|---|
| prefill 토큰/배치 | **256** 고정 |
| **제어 타깃 1: decode 개수** | **123** (= FFN 균형 batch 379 − prefill 256) |
| **제어 타깃 2: decode KV 합** | **12.3M** (= 123 × 100K) |
| prefill KV-work 타깃 | **25.6M** (= 256 토큰 × depth-합) |
| **AGE_CAP** | **2** (공정성: wait≥2 강제 포함, 대기 ≤3 batch) |
| 균형 시간 X | ~51 µs (TBT≈4.1ms) |
| 균형 ctx | ~100K (하드웨어 상수, *유도용* — 워크로드 강제 아님) |
| aggregate KV cap | **30M** (= 12.3M×2슬롯+여유, 5TB/80GB·stack). ※현 config 60M(512용) → 256 시 30M 로 조정 |
| (대안) prefill 512 | 모든 값 2배. FFN MFU 가 batch 379 에서 포화 *불가* 판명 시만 |

**길이분산 무관:** avg 100K 는 KV 캡 유도용 중간값일 뿐. former 는 avg 안 보고 **(개수 123, Σkv
12.3M) 두 타깃** 만 맞춤 → 어떤 길이분포(짧/긴/혼합)든 거대 풀서 조합으로 동작.

> ⚠ **밴드 [11.1M,13.5M] 는 former 입력이 *아님*** — S4 측정의 idle-SLA 진단 라벨(±10%→idle≤10%)일
> 뿐. former-v2 는 타깃(12.3M, 123)+age-cap 만 봄(steering 이 오버슈트 자체 방지, 밴드 가드 불요).

## 2. 배치 구성 알고리즘 (OPERATING_POINT.md §3 — 이걸 구현)

로컬 그리디 **steering + age-cap**:
```
한 μ-batch (decode):
  n=0, S=0
  while n < 123 and S < 12.3M and pool:
    if (wait ≥ AGE_CAP=2 인 요청 있음): 가장 오래된 그것 admit   # 공정성(강제)
    else: ideal=(12.3M − S)/(123 − n) 에 가장 가까운 디코더 admit  # steering
  나머지 대기 요청 wait += 1
prefill 도 동일(256 토큰, depth-합 25.6M). window=3 순차.
```
- closest-to-ideal 이 두 타깃 동시 수렴(자기보정) + 오버슈트 자체 방지(상한 가드 불필요).
- age-cap: pure steering 의 starvation(ideal-크기만 cherry-pick) 해소. 강제분은 steering 이 보정.
- 검증됨: 변종 풀 spread~1%, 서빙분포=arrival분포(starvation 0), 대기 ≤3.

## 3. 남은 구현

### former-v2 — `admission.layer1` 재작성 (먼저)
- 현 layer1 = "Σkv ≥ 목표까지 head-of-line admit"(개수 통제 없음). → **steering + age-cap 으로 교체.**
- 매 step `ideal=(target_kv−S)/(target_count−n)` 가장 가까운 디코더 선택, wait≥AGE_CAP 강제.
  **풀을 KV 길이로 인덱싱(버킷/정렬)** + 요청별 wait 추적.
- config 추가: `decode_count_target=123`, `prefill_chunk_default=256`, `age_cap=2`,
  `kv_capacity_aggregate` 60M→**30M**(256 기준). `kv_operating_target_tokens`=12.3M.
- prefill 도 depth-합 25.6M steering(256 토큰 분배).
- **반드시 유지:** 생애 전이(prefill_processed≥len(prompt) → PREFILL→DECODE), 빈슬롯 재충전 트리거.
- 변경별 타깃 테스트(test_admission*) 갱신 + 즉시 커밋.

### S4 — 측정 substrate
- **(a) `Request` prompt_len 경량화** — 현재 `prompt_tokens=[0]*num_prefill` materialize 가 실
  1M-ctx 트레이스서 **OOM**(scheduler 는 len 만 씀). prompt_len(int) 으로 바꿔 OOM 해소.
  (S3 의 유일 잔여 red `test_real_longbench` 가 이것 때문.)
- (b) `Request.first_token_time` + 완료요청 sink. L 도달 첫 decode 시 기록.
- (c) `debug_phase2/measure_steady.py` 작성(스텁 해소): TTFT=`first_token−arrival`,
  TBT=`(completion−first_token)/(decoded−1)`. p50/p90/max. warm-start seed 플래그(§2.6).
- (d) **idle ↔ 이론 일치 검증**(§5, 핵심 성공 판정): 스케일 스펙트럼(짧→B-bound / ~100K→idle≈0
  balanced / 긴1M→A-bound[PIM hero]). 실측 idle 이 이론(steering 으로 ~100K 면 ≈0)에 가까운지 = F2/F3 발현 증거.

### S5 — 데드코드 정리 + 문서
- S2/former-v2 orphan(deadband.py·instance_pipeline vestigial 등) 제거. `배치_생애.md`·README
  (Target Workload, HW 80GB/stack·5TB[256 기준]) 갱신. REPORT 에 측정 결과 + prefill sweep 정정
  ("512만 균형"은 측정오류, 실제 family).

## 4. 작업 습관 (이전 세션 실수 — 꼭 지켜라)
- **추측 말고 코드/측정으로 확인.** op-time 직접 산출. 가설은 프로토타입으로 검증.
- **★ 심볼 삭제 시 tests *및 src 호출처* 모두 grep** (S2 가 run.py 호출처 놓쳐 회귀 — S3 에서 수정).
- **도구 과병렬 금지**(cascade 취소). 무거운 측정/full-sim 은 백그라운드 하나씩. **풀 스위트 반복
  금지** — 변경별 타깃 테스트만(full-sim acceptance 는 ~5분 → 남발 금지).
- **변경 즉시 커밋**(heredoc 메시지), `PYTHONIOENCODING=utf-8` + 파일 출력(콘솔 cp949 깨짐).
- bash cwd 불안정 — `cd /c/Users/rhs02/Desktop/puls-rfc/implementation &&` 로 시작, 한글 경로는 Glob/Read.
- 깨진 옛 테스트는 풀 모델/steering 에 맞게 갱신(이유 주석).

## 5. 코드 지도
- `src/puls_sched/` — `main_loop.py`(SchedulerCore·ADMISSION_TICK·생애전이·_recompose_mb[backfill
  삭제됨]), `admission.py`(layer1 — **former-v2 대상**, balance/mfu 삭제됨), `dispatcher.py`(FFN
  노드·INSTANCE_B·num_gpus), `config.py`(compute_gpu/ffn_op_time_s num_gpus·default_dummy_config),
  `pim_emulator.py`(op_time=ns, k_aggregate=2048), `node.py`(NodeType 5: +FFN), `dag.py`(O_PROJ→FFN),
  `request.py`(FSM·**prompt_len 경량화 대상**), `run.py`(payload={} 수정됨), `evaluator.py`.
- `debug_phase2/` — `OPERATING_POINT.md`(★canonical), `PLAN.md`, `REPORT.md`, `STEP7_PROMPT.md`(이 파일),
  `sweep_operating_point.py`·`proto_steering.py`·`proto_steering_fair.py`(검증 코드).
- `tests/` — test_admission(21, former-v2 대상)·test_phase2_ffn_stage(9)·acceptance·cross_module 등.

시작: ARCHITECTURE.md + OPERATING_POINT.md + PLAN.md 정독 → former-v2(steering+age-cap)부터.
변경마다 타깃 테스트 + 즉시 커밋.
