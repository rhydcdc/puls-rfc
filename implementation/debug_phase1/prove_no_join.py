"""합류 부재 증명 — 캐파 압박 트레이스에서.

관측:
  (1) mb 다중 생성 여부 (캐파 초과 적체 → 새 mb)
  (2) 기존 mb 에 신규 요청 합류 여부 (= 생성 후 request_ids 증가하는 mb 존재?)

증명 논리: 각 mb 의 request_ids() 를 생성 시점에 기록하고, 이후 변화를 추적.
합류가 있으면 어떤 mb 의 request id 집합이 *커진다*. 없으면 생성 시 집합이 고정.
"""
import dataclasses

from puls_sched.event import EventType
from puls_sched.run import Run

run = Run.init(
    "puls_sched.config:default_dummy_config",
    "debug_phase1/data/trace_long_pressure.csv",
    "debug_phase1/_scratch_pressure",
)
sched = run.scheduler
cfg = run.config

# mb 생성 시점 request id 집합 기록
mb_birth_reqs = {}          # mb_id -> frozenset(req ids at creation)
orig_register = sched.dispatcher.register
def traced_register(mb):
    mb_birth_reqs[mb.id] = frozenset(mb.request_ids())
    return orig_register(mb)
sched.dispatcher.register = traced_register

MAX_STEPS = 3_000_000
max_concurrent_mb = 0
admit_ticks = 0
kv_peak = 0
for i in range(MAX_STEPS):
    if len(sched.queue) == 0:
        print(f"[queue empty at step {i}]")
        break
    nxt = sched.queue._heap[0][2].type if sched.queue._heap else None
    sched.step()
    if nxt == EventType.ADMISSION_TICK:
        admit_ticks += 1
    cur_mb = len(sched.window.current_ids())
    max_concurrent_mb = max(max_concurrent_mb, cur_mb)
    kv_peak = max(kv_peak, sched.kv_accountant.used)

# 합류 검사: 현재 살아있는 mb 들의 request_ids 가 birth 집합보다 커졌는가?
join_detected = False
grown = []
for mb_id, mb in sched.dispatcher.micro_batches.items():
    birth = mb_birth_reqs.get(mb_id, frozenset())
    now = frozenset(mb.request_ids())
    if not now.issubset(birth):     # 새 req 가 추가됨 = 합류
        join_detected = True
        grown.append((mb_id, sorted(birth), sorted(now)))

print("=== 캐파 압박 baseline (step cap", MAX_STEPS, ") ===")
print("총 mb 생성 수:", len(mb_birth_reqs))
print("동시 최대 mb (window):", max_concurrent_mb, "/ capacity", sched.window.capacity)
print("KV used peak:", kv_peak, "/ capacity", cfg.admission.kv_capacity_aggregate,
      f"({100*kv_peak/cfg.admission.kv_capacity_aggregate:.1f}%)")
print("ADMISSION_TICK 처리 수:", admit_ticks)
print("아직 처리 안 된 queue 길이:", len(sched.queue))
print("in_flight 요청 수:", len(sched.in_flight_requests))
print()
print(">>> 합류(기존 mb 에 신규 req 추가) 발생?:", join_detected)
if grown:
    for mb_id, b, n in grown[:5]:
        print(f"   mb {mb_id}: birth={b} -> now={n}")
else:
    print("   (no mb grew its req set after birth -- join path absent)")

# 왜 새 mb 도 안 생기나 — 큐 적체 + 캐파 점유 진단
print("\n=== 새 mb 미생성 원인 진단 ===")
print("queue 적체:", len(sched.queue.__dict__.get('_heap', [])), "(ADMISSION_TICK 포함)")
print("request_queue 적체:", len(sched.request_queue))
print("KV remaining:", sched.kv_accountant.remaining)
# 큐에 대기 중인 요청들의 최소 kv_length — 이게 remaining 보다 크면 영원히 admit 불가
pending_kv = [r.kv_length for r in sched.request_queue._q]
if pending_kv:
    print("대기 요청 최소 kv_length:", min(pending_kv), "/ remaining:", sched.kv_accountant.remaining,
          "-> admit 가능?", min(pending_kv) <= sched.kv_accountant.remaining)

# mb 생성 시점 집합 샘플
print("\nmb 생성 시점 request id 집합 (앞 12개):")
for mb_id in sorted(mb_birth_reqs)[:12]:
    print(f"  mb {mb_id}: {sorted(mb_birth_reqs[mb_id])}")
