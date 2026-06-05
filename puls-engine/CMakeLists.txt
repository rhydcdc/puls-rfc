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
)
foreach(t ${PULS_TESTS})
  add_executable(${t} validation/${t}.cpp)
  target_link_libraries(${t} PRIVATE puls_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
