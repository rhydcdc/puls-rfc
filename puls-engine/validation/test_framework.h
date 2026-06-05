// PULS-ENGINE — 외부 의존 0 미니 테스트 프레임워크. 각 test_*.cpp 가 int main() 을 갖는다.
// 사용: CHECK(cond, "msg"); CHECK_NEAR(a, b, tol, "msg"); return puls_test::summary();
#pragma once
#include <cstdio>
#include <cmath>
#include <string>

namespace puls_test {
inline int& failures() { static int f = 0; return f; }
inline int& checks()   { static int c = 0; return c; }

inline void record(bool ok, const char* expr, const char* msg, const char* file, int line) {
    ++checks();
    if (!ok) {
        ++failures();
        std::printf("  FAIL [%s:%d] %s  (%s)\n", file, line, msg, expr);
    }
}
inline int summary() {
    std::printf("%s — %d checks, %d failures\n",
                failures() == 0 ? "PASS" : "FAIL", checks(), failures());
    return failures() == 0 ? 0 : 1;
}
} // namespace puls_test

#define CHECK(cond, msg) \
    ::puls_test::record((cond), #cond, (msg), __FILE__, __LINE__)

#define CHECK_NEAR(a, b, tol, msg) \
    ::puls_test::record(std::fabs((double)(a) - (double)(b)) <= (tol), \
                        #a " ~= " #b, (msg), __FILE__, __LINE__)

#define CHECK_REL(a, b, rel, msg) \
    ::puls_test::record(std::fabs((double)(a) - (double)(b)) <= (rel) * std::fabs((double)(b)), \
                        #a " ~=rel " #b, (msg), __FILE__, __LINE__)
