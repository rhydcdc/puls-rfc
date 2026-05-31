"""mb 등록 타임라인 진단 — 합류 ON/OFF 에서 새 mb 가 언제(혹은 안) 생기나.

가설 A(조기종료) vs B(합류가 freed KV 를 backfill 로 잡아먹어 새 mb 미생성) 판별.
각 dispatcher.register 시점의 (step, clock, kv_used, queue_len, window) 기록.
drain 까지(또는 --cap) 돌려 mb_count·max_window·타임라인 출력.
"""
import argparse
import sys

sys.path.insert(0, "debug_phase1")
from puls_sched.run import Run

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--label", default="diag")
ap.add_argument("--no-join", action="store_true")
ap.add_argument("--cap", type=int, default=8_000_000)
ap.add_argument("--config", default="puls_sched.config:default_dummy_config")
ns = ap.parse_args()

run = Run.init(ns.config, ns.trace, f"debug_phase1/_scratch_diag_{ns.label}")
sched = run.scheduler
if ns.no_join:
    sched._try_join = lambda active_req_ids: set()

step_box = [0]
events = []
orig = sched.dispatcher.register
def traced(mb):
    events.append((
        step_box[0], sched.clock.now, sched.kv_accountant.used,
        len(sched.request_queue), len(sched.window.current_ids()), mb.id,
    ))
    return orig(mb)
sched.dispatcher.register = traced

PROGRESS_EVERY = 200_000
i = 0
max_window = 0
drained = False
while i < ns.cap:
    if (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
            and len(sched.in_flight_requests) == 0):
        drained = True
        break
    step_box[0] = i
    if not sched.step():
        break
    w = len(sched.window.current_ids())
    if w > max_window:
        max_window = w
    i += 1
    if i % PROGRESS_EVERY == 0:
        print(
            f"[progress] step={i} clock={sched.clock.now:.1f} "
            f"mb_count={len(events)} window={w} max_window={max_window} "
            f"kv_used={sched.kv_accountant.used} queue={len(sched.request_queue)} "
            f"in_flight={len(sched.in_flight_requests)}",
            flush=True,
        )

L = [f"=== mb timeline: {ns.label} join={'OFF' if ns.no_join else 'ON'} ==="]
L.append(f"trace={ns.trace} steps={i} drained={drained} "
         f"mb_count={len(events)} max_window={max_window}/{sched.window.capacity}")
for (st, ck, kv, q, win, mid) in events:
    L.append(f"  mb={mid} step~{st} clock={ck:.1f} kv_used={kv} queue={q} window_after={win}")
out = "\n".join(L)
print(out)
with open(f"debug_phase1/mbtl_{ns.label}.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
