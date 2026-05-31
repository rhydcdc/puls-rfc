"""합류 vs admission 경쟁 메커니즘 직접 검증 (가설 B) — 작은 트레이스로 수초 완주.

질문: 완료로 KV 가 풀리는 순간, 같은 완료 이벤트의 `_recompose_mb→_try_join`(즉시)이
다음 admission tick(+10µs) 보다 먼저 freed KV 를 기존 mb 에 backfill 하는가?
→ 그렇다면 합류 ON 에선 새 mb 가 안 생겨 mb_count 가 OFF 보다 작아야 한다.

작은 prefill(50)·decode(5) + KV 캐파 축소(250)로 backlog 강제. 트레이스 크기와
무관한 이벤트 순서 문제라, 빠른 트레이스로 동일하게 드러남.
"""
import sys

sys.path.insert(0, "debug_phase1")
from puls_sched.run import Run

TRACE = "debug_phase1/data/trace_tiny_race.csv"
KV_CAP = 250  # 6 req × kv 55 = 330 > 250 → ~4 admit, 2 backlog


def run_once(no_join: bool):
    run = Run.init("puls_sched.config:default_dummy_config", TRACE,
                   f"debug_phase1/_scratch_race_{'off' if no_join else 'on'}")
    sched = run.scheduler
    sched.kv_accountant._capacity = KV_CAP   # KV 캐파 축소 (can_admit / _try_join 둘 다 참조)
    if no_join:
        sched._try_join = lambda active_req_ids: set()

    timeline = []
    orig = sched.dispatcher.register
    def traced(mb):
        timeline.append((sched.clock.now, sched.kv_accountant.used,
                         len(sched.request_queue), mb.id))
        return orig(mb)
    sched.dispatcher.register = traced

    i = 0
    max_window = 0
    while i < 5_000_000:
        if (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
                and len(sched.in_flight_requests) == 0):
            break
        if not sched.step():
            break
        max_window = max(max_window, len(sched.window.current_ids()))
        i += 1
    return len(timeline), max_window, i, timeline


L = [f"=== join-vs-admission race (KV_CAP={KV_CAP}, trace={TRACE}) ==="]
for no_join in (True, False):
    mb_count, max_window, steps, timeline = run_once(no_join)
    L.append(f"\njoin={'OFF' if no_join else 'ON '}  mb_count={mb_count}  "
             f"max_window={max_window}  steps={steps}")
    for (ck, kv, q, mid) in timeline:
        L.append(f"    mb={mid} clock={ck:.2f} kv_used_at_register={kv} queue={q}")
out = "\n".join(L)
print(out)
with open("debug_phase1/race_result.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
