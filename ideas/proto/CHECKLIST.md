# Pre-Positioning 프로토타입 — 구현 체크리스트

> 근거 = [../predictive_prepositioning_scheduler.md](../predictive_prepositioning_scheduler.md). 전부 **C++17**. 계획서 외 로직 금지, 누락 금지.

## 0. 제약 (불변)
- [ ] 모든 코드는 `ideas/proto/` 안에만. **`puls-engine/` 등 밖은 절대 수정·침범 금지.**
- [ ] 외부 core 필요분 → `ideas/proto/core/` 에 **verbatim 복사**(무수정). 빌드는 `-Iideas/proto` 로 사본만 참조, 밖 `#include` 0.
- [ ] **폴더 분리 = 스케줄러 로직(`scheduler/`) · 시뮬레이터(`sim/`) · 검증(`validation/`)** (CONTRACT §2 원칙).
- [ ] 동작점 수치(100K·62·1.33TB…) 하드코딩 금지 — 전부 derive 산출.
- [ ] 모듈별 단위검증 필수 — **핵심만, 과잉 금지**(모듈당 ~4 체크).

## 1. 디렉터리 (역할별 분리)
```
ideas/proto/
  core/          # puls-engine/core/* 필요분 verbatim 복사 (무수정)
  scheduler/     # 신규 스케줄러 로직 — 순수 결정/상태, 워크로드·RNG 없음
    queue.*        # 글로벌 풀 + age-cap 강제 라우팅 결정
    cache.*        # KV 캐시 정책 (적격·admit·evict·3-tier 조회)
    preposition.*  # 결정론 pre-positioning composer (B 핵심 로직)
  sim/           # 시뮬레이터 — 워크로드·드라이버·측정 (sim 전용)
    workload_mt.*  # 멀티턴 워크로드
    harness.h      # 공용 하니스(동작점·엣지컷오프·복귀스케줄·시간모델) — A·B 비교 고정
    metrics.h      # 공용 측정 (header-only)
    baseline.cpp   # A 드라이버: 현 로컬 steering port
    prepo.cpp      # B 드라이버: 글로벌 pre-positioning
  validation/    # 검증 — 단위 테스트
    test_*.cpp
  REPORT.md      # 실측 대조
```
빌드: `g++ -std=c++17 -O2 -Iideas/proto ...` (사본 `core/` + `scheduler/` + `sim/`).
> 경계: `scheduler/` 는 분포 B·RNG·라운드 루프 미포함(순수 로직). 워크로드·churn·측정은 `sim/` 만.

## 2. core 복사 (verbatim, 무수정)
- [ ] spec.h · optime.{h,cpp} · operating_point.h · derive.{h,cpp}  ← 동작점 산출
- [ ] steering.{h,cpp}  ← A 비교군용
- [ ] request_source.h · workload.{h,cpp}  ← sample_distribution_b
- [ ] global_scheduler.{h,cpp}  ← gate, cold_start (A·B 공용)
- [ ] 복사 후 `core/` 만으로 빌드 smoke (밖 의존 0 확인)

## 3. 스케줄러 로직 (`scheduler/`, +검증)

### queue — 유한 큐 + 글로벌 age-cap
- [ ] 도착 타임스탬프 달린 **대용량 유한 큐**(요청 id·prompt·max_tokens).
- [ ] `pull_near(ideal, cap)` — 길이-fit 최근접 실 요청(없으면 -1).
- [ ] 글로벌 age-cap: 대기 > cap → **가장 비슷한 노드로 강제 배정 + 강제 카운트**.
- [ ] test_queue: ①age-cap 초과 강제·카운트 정확 ②길이-fit 최근접 ③FIFO(먼저 들어온 게 cap 내 빠짐) ④대용량 시 길이 고갈 0.

### cache — on-node KV 캐시 (노드별 + 글로벌 id 인덱스)
- [ ] 적격: 완료 길이 > **eligibility_threshold(스윕)** 만.
- [ ] capacity = **derive 잔여** `hbm_capacity_tb − instance_a_tb` (노드별, 하드코딩 0).
- [ ] admit: KV바이트 ≤ 잔여 → **무조건**(개수캡 없음). 넘으면 거부(하위계층行).
- [ ] evict: **evict_age(스윕, wall-clock idle)** 초과 방출.
- [ ] 3-tier 조회(id): HBM=hit(재로드0·P1재계산0) / SSD생존=miss(재로드=속도×바이트) / 완전방출=miss(P1 전체재계산). **P2 프리필·[P1+P2] attention 은 어느 경우든 항상 지불.**
- [ ] test_cache: ①잔여≥크기 admit·넘으면 거부 ②evict_age 초과 방출 ③적격임계 미만 비캐싱 ④hit/SSD/recompute 비용 분기 정확.

### preposition — 결정론 pre-positioning composer
- [ ] 입력 = 노드 현 디코더(각 잔여 = max_tokens−dec) + 동작점 → **이번/다음 라운드 빠질 슬롯 예측**.
- [ ] 빠질 슬롯마다 **두 배치 평균을 100K 로 되돌리는 정확한 길이** 산출 → `queue.pull_near` presend.
- [ ] 노드 steering·잉여·노드 힐링 **없음**(결정론 1회 선택).
- [ ] test_preposition: ①잔여로 완료 슬롯 정확 예측 ②presend 후 두 배치 (62, 6.15M) 재센터 ③off-fit 강제분 보정.

## 4. 시뮬레이터 (`sim/`)

### workload_mt — 멀티턴 워크로드 (sim 전용)
- [ ] 요청 = id + max_tokens(**결정론 decode 길이, EOS 무관**). 완료 = `dec == max_tokens` (jitter 0).
- [ ] 완료 후 **길이의존 복귀확률**(short↓/mid·long↑, 평균~2.5턴, continue≈0.6/end≈0.4) → 누적컨텍스트(이전+새메시지) 새 arrival 재진입.
- [ ] 신규 arrival = 분포 B + 엣지 게이팅.
- [ ] test_workload_mt: ①완료 정확히 max_tokens 스텝 ②복귀확률 short<long ③복귀 시 길이 누적 ④평균 턴수≈2.5.

### baseline.cpp (A — 현 로컬 steering port)
- [ ] 유한 큐 위에서 현 lifecycle 로직 port: **잉여 10 · 2 μ-batch steer_decode_set · 노드 age-cap · per-completion ideal=hole 힐링**.
- [ ] cold_start + gate (core 사본) 동일. **멀티턴 워크로드·SSD 오프로드는 동일 공유**(공정 대조). 단 **HBM 캐시 비활성**(현 설계 = "캐싱 없이 방출" → 복귀는 SSD 재로드/재계산) · 글로벌 pre-positioning·글로벌 age-cap 없음.
- [ ] metrics 출력.

### prepo.cpp (B — 글로벌 pre-positioning)
- [ ] 노드 = 실행기: **124(2×62) 상주, 노드 steering·잉여·노드 힐링 전부 없음**.
- [ ] cold_start: 글로벌 정밀 124 센터(100K) + gate.
- [ ] 정상상태: `scheduler/preposition` 결정론 presend(두 배치 항상 100K) + 글로벌 age-cap 강제 라우팅.
- [ ] on-node 캐시 + **멀티턴 복귀 시 캐시 보유 노드로 히트 라우팅**.
- [ ] metrics 출력.

### metrics.h (공용, header-only)
- [ ] Σdev(평균·최악·miss분류 count/dev·batch1/2) · p99 tail(시간단위 = optime `balance_time_us`×num_layers) · edge% · **글로벌 강제율** · **캐시 히트율**(evict_age·적격임계 함수→knee) · **누적 절약**(hit=P1 재로드+재계산 절약, reload=속도×길이) · 100K 유지(풀평균·상주 20/70/10).

## 5. 스윕 + REPORT.md
- [ ] 3축 스윕 → 각 knee: **eligibility_threshold · evict_age · global_age_cap**.
- [ ] **A vs B 동일 워크로드 대조표.**
- [ ] 연구질문: ①복귀 jitter 흡수 최소슬랙 ②글로벌 제어비용 vs forward-pass 박자(`balance_time_us` 대비) ③캐시 tail 개선.
- [ ] 실측 숫자만(추측 금지).

## 6. 에이전트 플랜 (총 7, 4 웨이브)
- **P0 — 나(직접):** 디렉터리(core/scheduler/sim/validation)·`core/` verbatim 복사·신규 헤더(queue.h/cache.h/preposition.h/workload_mt.h/metrics.h) **인터페이스 확정**(병렬 계약 고정) → core smoke 빌드.
- **P1 — 3 병렬(파일 disjoint, worktree 불요):**
  - A1 scheduler/queue.cpp + validation/test_queue
  - A2 scheduler/cache.cpp + validation/test_cache
  - A3 sim/workload_mt.cpp + validation/test_workload_mt
- **P2 — 2 병렬:**
  - A4 sim/baseline.cpp (A)
  - A5 scheduler/preposition.* + sim/prepo.cpp + validation/test_preposition (B)
- **P3 — 1:** 통합 빌드·3축 스윕 실행·REPORT.md(실측).
- **P4 — 1(검증 전담):** 계획서 대비 감사 — ①체크리스트 전항목 구현 ②추가 로직 0 ③`git status`로 `ideas/` 밖 변경 0 ④폴더 분리 준수 ⑤모듈 테스트 전부 통과 ⑥누락 0. 불일치 시 보고(재작업 지시).
