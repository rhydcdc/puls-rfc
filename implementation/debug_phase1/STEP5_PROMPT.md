# STEP 5 작업 프롬프트 (새 대화용)

아래 내용을 새 대화창에 그대로 붙여넣으세요.

---

PULS 스케줄러 Phase-1 디버깅의 STEP 5(재검증·스윕)를 진행한다. 수정 들어가기 전에
아래 문서들을 **꼼꼼히 다 읽고** 맥락을 완전히 파악한 뒤 시작해라. 그리고 너의 작업
습관에 대한 당부가 맨 아래 있으니 반드시 지켜라.

## 먼저 읽을 것 (순서대로)

1. `README.md` / `ARCHITECTURE.md` — PULS 아키텍처 전체 (HBM-PIM, instance disaggregation,
   SP-PIM, scheduler, balance 4요소). 무엇을 만드는 프로젝트인지.
2. `배치_생애.md` (repo 최상위) — 배치(마이크로배치) 생애의 핵심 설계. 세 한계(KV캐파/
   배치크기/window), 양방향 합류, 게이트(유휴율), 종료 조건. **STEP 5의 기준 문서.**
3. `implementation/debug_phase1/PLAN.md` — 디버깅 계획·체크리스트. STEP 1~4 완료, STEP 5 항목.
4. `implementation/debug_phase1/REPORT_baseline.md` — baseline 관측·근본 원인·수정 설계·
   진행 결과 전부 (§1~§11). 특히 §8(세 한계), §10(admission tick 버그), §11(before 수치).
5. 핵심 소스: `implementation/src/puls_sched/` 의 `admission.py`(layer1, balance_*, 
   _try_join 은 main_loop), `main_loop.py`(_recompose_mb, _try_join, evict, event-driven
   admission), `config.py`(AdmissionConfig), `idle_telemetry.py`, `evaluator.py`.

## 지금까지 한 것의 의의 (STEP 1~4, 모두 커밋·푸시 완료)

문제: 실 트레이스에서 PIM idle 99.66%. 원인은 (a) 트레이스 성질(decode 350 고정) +
(b) balance 4요소 미발현. 디버깅으로 밝힌 근본 원인과 수정:

- **STEP 1** — admission 이 한 tick 에 가용 요청을 KV 캐파까지 한 mb 로 몰아넣어 단일 mb
  독점·직렬 처리 → **배치 크기(seq) 상한**을 KV 캐파와 분리 신설. mb 다중화 달성.
- **STEP 2.5** (측정 중 발견) — admission tick 이 고정 10µs 타이머로 self-reschedule 되어
  GPU 가 긴 op 도는 동안 헛돌아 step·메모리 폭증(3M step 미완주→8.9GB). **event-driven
  admission**(완료/도착 시에만)으로 전환. light_pressure 3M+→76,820 step 완주(40× 가속).
- **STEP 3** — **양방향 합류**: 진행 중 mb 에 큐의 신규 요청을 backfill. 게이트 = 유휴율
  (gpu_idle>θ 면 prefill, pim_idle>θ 면 decode; 양쪽 포화면 닫힘). prefill/decode 분류는
  _populate_mb_phases 가 prompt 유무로 자동. evict = "완료 AND 합류 불가". 무실효였던
  balance_intra_A 의 decode +1 제거(이제 _try_join 이 decode 전담).
- **STEP 4** — 회귀 49 passed(lifecycle 17 실트레이스·stress·e2e·acceptance). 사전버그
  7개(낡은 lifecycle 테스트)도 수정.

→ **확정: 양방향 합류까지 구현+검증 완료. 배치_생애.md 와 코드 정합.** 아직 idle 이
실제로 떨어지는지(목적 달성)는 **STEP 5 에서 측정**한다.

## STEP 5 에서 할 일 (PLAN STEP 5 참조)

핵심 질문: **수정 후 정말 idle 이 떨어지는가? (양방향 합류·세 한계 분리의 효과 입증)**

1. **before/after idle 측정** — 합성 트레이스(`implementation/debug_phase1/data/` 의
   T-S/T-L/T-M)로:
   - T-L(long, ctx>56K) — GPU/종합 idle 감소 확인 (PIM bound 구간에 prefill 합류)
   - T-M(혼합 3:7) — 양방향 idle 동시 감소
   - T-S(short, 대조군) — idle **불변**이어야 정상 (과잉 수정 방지). short 는 packing 으로
     PIM idle 이 구조적 — 안 떨어지는 게 맞음.
   - 주 지표 = `idle_fraction["pim_instance_a"]` (record_active 기반). `pim_utilization`은
     dispatch 간격 적산이라 신뢰 X.
2. **스윕** — 배치 크기 {256, 512}, 합류 게이트 θ_high {0.3, 0.1}.
   - θ_high: 목적이 "유휴율 0 수렴"이라 30%는 느슨, 10%는 조임(진동 위험). 실측 비교.
3. **hysteresis 부활** — `idle_theta_low`(현재 0.1, **미사용 placeholder** — 로직 0개)를
   살려 이중 임계(θ_high 초과 시 합류, θ_low 미만 시 멈춤). θ_high 조일 때 진동 방지.
4. **token budget closed-form** — 모델 미실행, 트레이스 decode/prefill 로 산출.
5. **배치_생애.md 갱신** — 스윕 확정값(배치 크기 상한, θ_high+θ_low)을 §세한계/§5 에 명시.
6. **before/after 보고서** 작성.

## 중요한 사전 지식 (헷갈리기 쉬운 것)

- **임계 컨텍스트 ≈ 56K**: 순수 decode 의 PIM/GPU = ctx/56,160. ctx>56K 면 PIM bound,
  미만이면 GPU bound. 합성 트레이스가 이 경계를 의도적으로 가르도록 설계됨.
- **decode 합류 비용**: decode 도 GPU 에서 QKV+O_PROJ 함. N 늘면 GPU 도 N 비례 증가 →
  ctx>56K 일 때만 PIM 이 GPU 추월. (배치_생애.md 참조)
- **pim_slack 0.9 마진**: balance_pim_slack 이 t_pim×0.9 로 GPU chunk 산출(PIM 을
  compute-bound 뒤 은닉). 살아있음, 유지.
- **idle 측정 분모 = wall-clock span**: 도착 성기면 빈 시간이 idle 오염 → 도착 포화 필요.
- **idle_theta_low 는 현재 안 씀**(미사용), θ_high(0.3)만 게이트에 사용 중.

## 측정·회귀 운영 주의 (시간 관리)

- **decode×80층이 step 의 지배 요인.** idle 측정 완주는 무겁다(트레이스당 수~수십 분).
  T-L 하나 먼저 백그라운드 → 결과 보고 T-M/T-S. decode 길이·도착률로 완주 시간 관리.
- **회귀는 "개발 중 가벼운 타깃만, 커밋 직전 풀 1회"**. 풀 회귀(lifecycle 실트레이스+
  stress)는 ~30분. 매번 다 돌리지 마라.
- 측정 스크립트는 `debug_phase1/measure_idle.py`(있음) 참고. 결과는 파일로 저장(콘솔
  cp949 가 유니코드 깨뜨림 — PYTHONIOENCODING=utf-8 또는 파일 출력).
- 패키지는 editable install 됨(`pip install -e .` 완료). `python -m pytest` 직접 가능.

## 작업 습관 당부 (꼭 지켜라)

- **한 메시지에 도구를 과하게 병렬로 띄우지 마라.** 하나 실패하면 나머지가 연쇄 취소된다.
  순차로, 하나씩.
- **git 조작 신중히.** 변경하면 **바로 커밋**해라(stash/worktree 남발 금지 — 이전에
  uncommitted evict 변경을 stash drop 으로 날린 적 있다). 커밋된 건 안전, uncommitted 만 위험.
- **추측하지 말고 측정해라.** baseline 에서 가설(ctx>56K PIM bound, 캐파초과→새mb,
  라이브락)이 세 번 빗나갔다. 코드·실측으로 확인.
- **테스트 실행 전 무엇을 검증하는지 먼저 말해라.**
- **PowerShell here-string(`@' '@`)에서 `->`·유니코드 깨짐 주의.** 커밋 메시지는 bash
  heredoc 또는 -F 파일 사용.
- 수정 후 의도 정합 여부를 배치_생애.md 기준으로 자가검증해라.

현재 상태: STEP 1~4 전부 커밋·푸시·origin/main 동기화 완료. working tree 깨끗
(untracked analysis/ · idle_long_pressure.txt · measure_idle.py 는 무관).
시작 전 위 문서들 다 읽고, STEP 5 접근 계획을 먼저 제시해라.
