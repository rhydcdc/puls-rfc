import sys
sys.path.insert(0, "debug_phase1")
from puls_sched.run import Run

run = Run.init(
    "config_small_cap:light_pressure_config",
    "debug_phase1/data/trace_light_pressure.csv",
    "debug_phase1/_scratch_diag",
)
sched = run.scheduler
lines = ["step | queue | in_flight | clock"]
LIMIT = 3_000_000
PROBE = 300_000
i = 0
while i < LIMIT:
    if (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
            and len(sched.in_flight_requests) == 0):
        lines.append(f"DRAINED at step {i}")
        break
    if not sched.step():
        lines.append(f"STOPPED at step {i}")
        break
    i += 1
    if i % PROBE == 0:
        lines.append(f"{i} | {len(sched.queue)} | {len(sched.in_flight_requests)} | clock={sched.clock.now:.1f}")
if i >= LIMIT:
    lines.append(f"HIT LIMIT {LIMIT} (not drained)")
out = "\n".join(lines)
print(out)
open("debug_phase1/diag_livelock_result.txt", "w", encoding="utf-8").write(out + "\n")
