"""단일 mb 귀결의 코드 경로 정밀 특정.

가설: 4 요청이 첫 ADMISSION_TICK 에 KVcap(4M) 내 전부 admit → 한 mb 로 묶임.
이후 request_queue 빈 채 → _schedule_next_admission_tick 가 self-push 는 하나
admit 대상 0 → 새 mb 생성 0. 기존 mb 만 _recompose_mb 반복.

검증: SchedulerCore 를 직접 구동하며 ADMISSION_TICK 마다 상태를 계측.
"""
import dataclasses

from puls_sched.config import default_dummy_config
from puls_sched.run import Run

# 작은 trace 로 init (실행은 직접 step 제어)
run = Run.init(
    "puls_sched.config:default_dummy_config",
    "debug_phase1/data/trace_long_min.csv",
    "debug_phase1/_scratch_single_mb",
)
sched = run.scheduler
kv = sched.kv_accountant
cfg = run.config

print("=== 초기 상태 ===")
print("KV capacity:", cfg.admission.kv_capacity_aggregate)
print("window capacity:", sched.window.capacity)
print("tick_interval_us:", cfg.admission.tick_interval_us)
print("trace 요청 kv_length 합 추정:")
# replay 로 요청 kv 확인
from puls_sched.trace import TraceReplayer
rp = TraceReplayer.load("debug_phase1/data/trace_long_min.csv")
total_kv = 0
for r in rp.replay():
    print(f"  req {r.id}: prefill={len(r.prompt_tokens)} kv_length={r.kv_length} arrival={r.arrival_time:.2f}")
    total_kv += r.kv_length
print("  Σ kv_length =", total_kv, "<= KVcap?", total_kv <= cfg.admission.kv_capacity_aggregate)

# step 을 돌리되 mb 생성·admission tick·queue 상태를 계측
mb_created = []
tick_count = 0
admit_events = []

orig_register = sched.dispatcher.register
def traced_register(mb):
    mb_created.append((sched.clock.now, mb.id,
                       len(mb.prefill_chunk), len(mb.decode_tokens)))
    return orig_register(mb)
sched.dispatcher.register = traced_register

# 제한된 step 수만 — 단일 mb 귀결은 초반에 결정됨
MAX_STEPS = 200_000
seen_admit_tick = 0
from puls_sched.event import EventType
for i in range(MAX_STEPS):
    if len(sched.queue) == 0:
        print(f"\n[queue empty at step {i}]")
        break
    ts = sched.queue.peek_timestamp()
    # peek 다음 event type 보려면 pop 전 heap 확인 — 간단히 step 후 상태 계측
    nxt = sched.queue._heap[0][2].type if sched.queue._heap else None
    sched.step()
    if nxt == EventType.ADMISSION_TICK:
        seen_admit_tick += 1
        if seen_admit_tick <= 8 or seen_admit_tick % 100000 == 0:
            admit_events.append((
                seen_admit_tick, round(sched.clock.now, 2),
                len(sched.request_queue), len(sched.in_flight_requests),
                len(sched.window.current_ids()), kv.used if hasattr(kv, "used") else "?",
            ))

print("\n=== mb 생성 이력 (register 호출) ===")
for t, mid, npf, ndec in mb_created[:20]:
    print(f"  ts={t:.2f} mb_id={mid} prefill_reqs={npf} decode_reqs={ndec}")
print("총 mb 생성 수:", len(mb_created))

print("\n=== ADMISSION_TICK 초반 계측 (tick#, clock, |queue|, |in_flight|, |window|, kv.used) ===")
for row in admit_events:
    print(" ", row)

print("\n=== KVAccountant 상태 ===")
print("  used:", getattr(kv, "used", "n/a"), "/ capacity:", getattr(kv, "capacity", "n/a"))
