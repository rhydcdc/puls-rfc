"""직렬 검증 (가속판) — 작은 캐파 + tiny 트레이스로 완주.

5 요청(각 kv 60K) / 캐파 200K → 첫 mb 에 3개 admit(180K), 2개 대기.
decode=2 로 첫 mb 빨리 완료 → 2번째 mb 생성/합류 여부 관측.
"""
import sys

from puls_sched.event import EventType
from puls_sched.run import Run

# debug_phase1 import 가능하도록
sys.path.insert(0, "debug_phase1")
from config_small_cap import small_cap_config  # noqa

run = Run.init(
    "config_small_cap:small_cap_config",
    "debug_phase1/data/trace_serial_tiny.csv",
    "debug_phase1/_scratch_serial_fast",
)
sched = run.scheduler

mb_log = []
orig_register = sched.dispatcher.register
def traced_register(mb):
    mb_log.append((round(sched.clock.now, 2), mb.id,
                   sorted(mb.request_ids()), len(sched.window.current_ids())))
    return orig_register(mb)
sched.dispatcher.register = traced_register

max_window = 0
i = 0
LIMIT = 50_000_000
while i < LIMIT:
    if (len(sched.queue) == 0
            and len(sched.window.current_ids()) == 0
            and len(sched.in_flight_requests) == 0):
        drained = True
        break
    if not sched.step():
        break
    max_window = max(max_window, len(sched.window.current_ids()))
    i += 1
else:
    drained = False

lines = []
lines.append("=== 직렬 검증 (가속판, cap=200K, 5 reqs x 60K) ===")
lines.append(f"완전 drain: {drained}, 총 step: {i}")
lines.append(f"총 mb 생성 수: {len(mb_log)}")
lines.append(f"동시 최대 window: {max_window} / capacity {sched.window.capacity}")
lines.append("")
lines.append("mb register 이력 (ts, mb_id, req_ids, window_at_register):")
for row in mb_log:
    lines.append(f"   ts={row[0]} mb={row[1]} reqs={row[2]} window={row[3]}")
lines.append("")
if len(mb_log) >= 2:
    serial = all(r[3] <= 1 for r in mb_log)
    lines.append("판정:")
    lines.append(f"  - 2번째 mb 생성됨: True (대기 요청이 결국 처리됨)")
    lines.append(f"  - 동시 다중 mb: {max_window > 1} (max window={max_window})")
    lines.append(f"  - 직렬(매 mb 생성 시 window<=1): {serial}")
    if serial:
        lines.append("  => 직렬 확증: 앞 mb 완료 후에야 다음 mb 생성, 동시 공존 없음")
elif len(mb_log) == 1:
    lines.append("판정: mb 1개만 — 대기 요청이 합류했거나 미처리. req_ids 로 확인:")
    lines.append(f"   mb0 최종 req 수 추적 필요")

out = "\n".join(lines)
print(out)
with open("debug_phase1/serial_result.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
