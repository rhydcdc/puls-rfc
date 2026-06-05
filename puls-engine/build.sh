#!/usr/bin/env bash
# PULS-ENGINE 빌드·테스트 — cmake 부재 환경용(g++ 직접). cmake 있으면 CMakeLists.txt 사용.
#   사용: bash puls-engine/build.sh
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}
FLAGS="-std=c++17 -O2 -I. -Ivalidation"
CORE="core/optime.cpp core/derive.cpp core/steering.cpp core/node_scheduler.cpp core/global_scheduler.cpp core/workload.cpp"
TESTS="test_optime test_derive test_steering test_node_scheduler test_global_scheduler test_integration test_meta test_lifecycle"

mkdir -p build
echo "== building core + tests =="
for t in $TESTS; do
  $CXX $FLAGS validation/$t.cpp $CORE -o build/$t.exe
done
echo "== building drivers =="
$CXX $FLAGS runtime/main_runtime.cpp runtime/runtime.cpp $CORE -o build/puls_runtime.exe
$CXX $FLAGS sim/sim.cpp $CORE -o build/puls_sim.exe
$CXX $FLAGS sim/lifecycle.cpp $CORE -o build/puls_lifecycle.exe
$CXX $FLAGS analysis/max_model.cpp $CORE -o build/max_model.exe

echo "== running tests =="
fail=0
for t in $TESTS; do
  echo "-- $t --"
  ./build/$t.exe || fail=1
done
if [ $fail -eq 0 ]; then echo "ALL TESTS PASS"; else echo "SOME TESTS FAILED"; exit 1; fi
