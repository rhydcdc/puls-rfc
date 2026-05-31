"""STEP 1 검증 — seq 상한이 mb 다중화를 만드는가.

same 트레이스(trace_serial_tiny: 5 req × kv 60K)를 두 config 로:
  (A) small_cap_config        — KV 200K, seq 무제한(256) → 직렬 기대 (max window=1)
  (B) small_cap_seqlimit_config — KV 200K, seq=2        → 다중 기대 (max window>1)
변수 하나(seq 상한)만 바꿔 인과 분리.
"""
import sys

sys.path.insert(0, "debug_phase1")

from puls_sched.run import Run


def run_one(config_factory, label):
    run = Run.init(
        f"config_small_cap:{config_factory}",
        "debug_phase1/data/trace_serial_tiny.csv",
        f"debug_phase1/_scratch_mux_{label}",
    )
    sched = run.scheduler
    mb_births = []
    orig = sched.dispatcher.register
    def traced(mb):
        mb_births.append((round(sched.clock.now, 1), mb.id,
                          sorted(mb.request_ids()), len(sched.window.current_ids())))
        return orig(mb)
    sched.dispatcher.register = traced

    max_window = 0
    i = 0
    LIMIT = 50_000_000
    while i < LIMIT:
        if (len(sched.queue) == 0
                and len(sched.window.current_ids()) == 0
                and len(sched.in_flight_requests) == 0):
            break
        if not sched.step():
            break
        max_window = max(max_window, len(sched.window.current_ids()))
        i += 1
    return mb_births, max_window, i


lines = []
for factory, label in [("small_cap_config", "A_seq_unlimited"),
                       ("small_cap_seqlimit_config", "B_seq2")]:
    births, mw, steps = run_one(factory, label)
    lines.append(f"=== {label} ({factory}) ===")
    lines.append(f"  총 mb 생성: {len(births)}, 동시 최대 window: {mw}, steps: {steps}")
    for b in births[:8]:
        lines.append(f"    ts={b[0]} mb={b[1]} reqs={b[2]} window_at_birth={b[3]}")
    lines.append("")

lines.append("판정:")
lines.append("  seq 무제한 → 직렬(window=1), seq=2 → 다중(window>1) 이면 STEP 1 성공")

out = "\n".join(lines)
print(out)
with open("debug_phase1/multiplex_result.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
