// PULS-ENGINE — 실 서빙 런타임 진입점. CONTRACT.md §2.
// main() 은 여기에만 둔다(runtime.cpp 에는 정의·노출 함수만 — 다중 main 금지).
// 헤더 새 파일을 만들지 않으므로, runtime.cpp 가 노출하는 단일 함수만 전방 extern 선언해 호출한다.

namespace puls {
void run_runtime_demo();  // 정의: runtime/runtime.cpp
}

int main() {
    puls::run_runtime_demo();
    return 0;
}
