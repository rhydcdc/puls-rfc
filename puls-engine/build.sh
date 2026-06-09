#!/usr/bin/env bash
# PULS-ENGINE 빌드·테스트 — cmake 부재 환경용(g++ 직접). cmake 있으면 CMakeLists.txt 사용.
#   사용: bash puls-engine/build.sh
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}
FLAGS="-std=c++17 -O2 -I. -Ivalidation"
CORE="core/optime.cpp core/derive.cpp core/steering.cpp core/node_scheduler.cpp core/global_scheduler.cpp core/workload.cpp"
# 흡수된 클러스터 레이어(C): 글로벌 age-cap 큐 + 멀티턴 캐시 + pre-position + 멀티턴 워크로드.
CLUSTER="scheduler/queue.cpp scheduler/cache.cpp scheduler/preposition.cpp sim/workload_mt.cpp"
TESTS="test_optime test_derive test_steering test_node_scheduler test_global_scheduler test_integration test_meta test_lifecycle test_efficiency"
# 클러스터 레이어 검증(C): 큐 강제 라우팅 · 3-tier 캐시 · 컨텐션 TBT · pre-position · 멀티턴 워크로드.
CLUSTER_TESTS="test_queue test_cache test_kpi test_preposition test_workload_mt"

mkdir -p build
echo "== building core + tests =="
for t in $TESTS; do
  $CXX $FLAGS validation/$t.cpp $CORE -o build/$t.exe
done
echo "== building cluster tests =="
for t in $CLUSTER_TESTS; do
  $CXX $FLAGS validation/$t.cpp $CLUSTER $CORE -o build/$t.exe
done
echo "== building drivers =="
$CXX $FLAGS runtime/main_runtime.cpp runtime/runtime.cpp $CORE -o build/puls_runtime.exe
$CXX $FLAGS sim/sim.cpp $CORE -o build/puls_sim.exe
$CXX $FLAGS sim/lifecycle.cpp $CORE -o build/puls_lifecycle.exe
$CXX $FLAGS analysis/max_model.cpp $CORE -o build/max_model.exe
$CXX $FLAGS analysis/efficiency.cpp $CORE -o build/efficiency.exe
# 클러스터 드라이버(C 채택 / A·B 대조군) — derive 동작점 위에서 실 KPI 산출.
$CXX $FLAGS sim/csched.cpp   $CLUSTER $CORE -o build/csched.exe
$CXX $FLAGS sim/baseline.cpp $CLUSTER $CORE -o build/baseline.exe
$CXX $FLAGS sim/prepo.cpp    $CLUSTER $CORE -o build/prepo.exe

echo "== running tests =="
fail=0
for t in $TESTS $CLUSTER_TESTS; do
  echo "-- $t --"
  ./build/$t.exe || fail=1
done
if [ $fail -eq 0 ]; then echo "ALL TESTS PASS"; else echo "SOME TESTS FAILED"; exit 1; fi
