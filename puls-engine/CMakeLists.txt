cmake_minimum_required(VERSION 3.16)
project(puls_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ── core: 공유 스케줄링 로직 (CONTRACT.md §3) ───────────────────────────────
add_library(puls_core STATIC
  core/optime.cpp
  core/derive.cpp
  core/steering.cpp
  core/node_scheduler.cpp
  core/global_scheduler.cpp
  core/workload.cpp        # sim/validation 전용 출처지만 core 라이브러리에 동봉(링크 편의)
)
target_include_directories(puls_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ── 흡수된 클러스터 레이어(C): 글로벌 age-cap 큐 + 멀티턴 3-tier 캐시 + pre-position + 멀티턴 워크로드 ──
add_library(puls_cluster STATIC
  scheduler/queue.cpp
  scheduler/cache.cpp
  scheduler/preposition.cpp
  sim/workload_mt.cpp
)
target_link_libraries(puls_cluster PUBLIC puls_core)

# ── 실 서빙 런타임 드라이버 ─────────────────────────────────────────────────
add_executable(puls_runtime runtime/main_runtime.cpp runtime/runtime.cpp)
target_link_libraries(puls_runtime PRIVATE puls_core)

# ── 시뮬레이터 드라이버 ─────────────────────────────────────────────────────
add_executable(puls_sim sim/sim.cpp)
target_link_libraries(puls_sim PRIVATE puls_core)

# 통합 lifecycle 시뮬 (프리필→디코드 종속성 + Σdev 측정)
add_executable(puls_lifecycle sim/lifecycle.cpp)
target_link_libraries(puls_lifecycle PRIVATE puls_core)

# 분석: 칩별 최대 모델 / SOTA 적합성 (derive 기반)
add_executable(puls_max_model analysis/max_model.cpp)
target_link_libraries(puls_max_model PRIVATE puls_core)

# 분석: 아키텍처 효율 (동작점 idle/pipeline-eff/PIM-숨음/균형, 일반화)
add_executable(puls_efficiency analysis/efficiency.cpp)
target_link_libraries(puls_efficiency PRIVATE puls_core)

# 클러스터 드라이버: C(채택) / A(ablation) / B(pre-position 대조군) — derive 동작점 위 실 KPI.
foreach(d csched baseline prepo)
  add_executable(${d} sim/${d}.cpp)
  target_link_libraries(${d} PRIVATE puls_cluster)
endforeach()

# ── 검증 (모듈별 + 통합) — ctest ────────────────────────────────────────────
enable_testing()
set(PULS_TESTS
  test_optime
  test_derive
  test_steering
  test_node_scheduler
  test_global_scheduler
  test_integration
  test_meta
  test_lifecycle
  test_efficiency
)
foreach(t ${PULS_TESTS})
  add_executable(${t} validation/${t}.cpp)
  target_link_libraries(${t} PRIVATE puls_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# 클러스터 레이어 검증: 큐 강제 라우팅 · 3-tier 캐시 · 컨텐션 TBT · pre-position · 멀티턴 워크로드.
set(PULS_CLUSTER_TESTS
  test_queue
  test_cache
  test_kpi
  test_preposition
  test_workload_mt
)
foreach(t ${PULS_CLUSTER_TESTS})
  add_executable(${t} validation/${t}.cpp)
  target_link_libraries(${t} PRIVATE puls_cluster)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
