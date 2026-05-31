"""Baseline dispatch_trace 분석 — balance 미발현·idle 구조 확증."""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "debug_phase1/baseline_long_min/report.json"
d = json.load(open(path))
ev = d["dispatch_trace"]
print("total dispatch events:", len(ev))

NT = {1: "QKV", 2: "PREFILL_ATTN", 3: "DECODE_ATTN", 4: "O_PROJ"}
c = Counter((e["resource"], NT.get(e["node_type"], e["node_type"])) for e in ev)
for k, v in sorted(c.items()):
    print("  ", k, v)

mbs = sorted(set(e["micro_batch_id"] for e in ev))
print("distinct micro_batch ids:", mbs, "count =", len(mbs))

last_ts = ev[-1]["timestamp"]
pim = [e for e in ev if e["resource"] == "PIM"]
pattn = [e for e in ev if e["node_type"] == 2]
print("last overall ts:", round(last_ts, 1))
print("PIM dispatches:", len(pim))
print("PREFILL_ATTN dispatches:", len(pattn),
      "last ts:", round(pattn[-1]["timestamp"], 1) if pattn else None)

# 각 mb에 몇 개 요청이 들어있었나 — dag_state_snapshot로는 알 수 없으니
# PREFILL_ATTN이 언제 끊기는지(=prefill 소진=순수 decode 전락 시점) 확인
B = 10
bp = [0] * B
bd = [0] * B
for e in ev:
    idx = min(B - 1, int(e["timestamp"] / last_ts * B)) if last_ts > 0 else 0
    if e["node_type"] == 2:
        bp[idx] += 1
    elif e["resource"] == "PIM":
        bd[idx] += 1
print("\n구간별 dispatch (시간 10등분):  bin | PREFILL_ATTN | PIM_DECODE")
for i in range(B):
    print(f"  {i:3d} | {bp[i]:13d} | {bd[i]:11d}")

print("\nPIM 첫 dispatch ts =", round(pim[0]['timestamp'], 1),
      f"({100*pim[0]['timestamp']/last_ts:.2f}% 지점)")
print("PREFILL_ATTN 마지막 ts =", round(pattn[-1]['timestamp'], 1),
      f"({100*pattn[-1]['timestamp']/last_ts:.2f}% 지점)")
