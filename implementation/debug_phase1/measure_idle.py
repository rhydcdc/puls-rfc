"""STEP 2.5 후 idle 측정 — STEP 3 합류 전 before 수치.

인자: config_module, trace_csv, label
완주 여부 + idle 3-key + mb 다중화(동시 최대 window) + 총 mb 수 측정.
"""
import sys
sys.path.insert(0, "debug_phase1")
from puls_sched.run import Run

cfg = sys.argv[1] if len(sys.argv) > 1 else "puls_sched.config:default_dummy_config"
trace = sys.argv[2] if len(sys.argv) > 2 else "debug_phase1/data/trace_long_pressure.csv"
label = sys.argv[3] if len(sys.argv) > 3 else "long_pressure"

run = Run.init(cfg, trace, f"debug_phase1/_scratch_m_{label}")
sched = run.scheduler

mb_count = 0
orig = sched.dispatcher.register
def traced(mb):
    global mb_count
    mb_count += 1
    return orig(mb)
sched.dispatcher.register = traced

max_window = 0
i = 0
LIMIT = 100_000_000
while i < LIMIT:
    if (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
            and len(sched.in_flight_requests) == 0):
        drained = True
        break
    if not sched.step():
        break
    w = len(sched.window.current_ids())
    if w > max_window:
        max_window = w
    i += 1
else:
    drained = False

tel = sched.admission.idle_telemetry
L = []
L.append(f"=== idle 측정: {label} ===")
L.append(f"config={cfg}")
L.append(f"trace={trace}")
L.append(f"drain={drained} steps={i} mb_count={mb_count} max_window={max_window}/{sched.window.capacity}")
L.append(f"idle gpu_a={tel.gpu_idle_fraction():.4f} pim_a={tel.pim_idle_fraction():.4f} gpu_b={tel.gpu_instance_b_idle_fraction():.4f}")
out = "\n".join(L)
print(out)
open(f"debug_phase1/idle_{label}.txt", "w", encoding="utf-8").write(out + "\n")
