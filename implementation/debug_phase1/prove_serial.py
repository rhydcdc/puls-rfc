"""직렬 처리 검증 — 첫 mb 완료 후 다음 mb 가 생기는가?

가설: 첫 mb(캐파 독점)가 전부 완료 -> KV release -> 다음 admission tick 에서 큐의
요청 admit -> 새 mb 생성. 단 동시 다중 mb 아니라 *순차*. window cap 3 무용.

관측: mb register 시각 + 그 순간 동시 살아있는 mb 수(window) + 완료 순서.
"""
from puls_sched.event import EventType
from puls_sched.run import Run

run = Run.init(
    "puls_sched.config:default_dummy_config",
    "debug_phase1/data/trace_long_pressure.csv",
    "debug_phase1/_scratch_serial",
)
sched = run.scheduler
cfg = run.config

mb_log = []   # (register_ts, mb_id, n_reqs, window_size_at_register)
orig_register = sched.dispatcher.register
def traced_register(mb):
    mb_log.append((round(sched.clock.now, 1), mb.id, len(mb.request_ids()),
                   len(sched.window.current_ids())))
    return orig_register(mb)
sched.dispatcher.register = traced_register

# 완료까지 — cap 넉넉히. window 동시성 최대치 추적.
MAX_STEPS = 80_000_000
max_window = 0
i = 0
while i < MAX_STEPS:
    if (len(sched.queue) == 0
            and len(sched.window.current_ids()) == 0
            and len(sched.in_flight_requests) == 0):
        print(f"[fully drained at step {i}]")
        break
    if not sched.step():
        break
    max_window = max(max_window, len(sched.window.current_ids()))
    i += 1

lines = []
lines.append("=== 직렬 처리 검증 ===")
lines.append(f"총 step: {i}")
lines.append(f"총 mb 생성 수: {len(mb_log)}")
lines.append(f"동시 최대 window (전 구간): {max_window} / capacity {sched.window.capacity}")
lines.append(f"남은 queue: {len(sched.queue)} in_flight: {len(sched.in_flight_requests)} "
             f"window: {len(sched.window.current_ids())}")
lines.append("")
lines.append("mb register 이력 (ts, mb_id, n_reqs, window_at_register):")
for row in mb_log:
    lines.append(f"   {row}")
if len(mb_log) >= 2:
    lines.append("")
    lines.append("판정:")
    lines.append(f"  - mb 다중 생성?: {len(mb_log) > 1}")
    lines.append(f"  - 동시 다중 mb?: {max_window > 1} (max window={max_window})")
    lines.append(f"  - 직렬(생성 시 window<=1)?: {all(r[3] <= 1 for r in mb_log)}")
out = "\n".join(lines)
print(out)
with open("debug_phase1/serial_result.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
