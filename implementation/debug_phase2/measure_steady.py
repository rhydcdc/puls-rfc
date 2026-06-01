"""Phase-2 정상상태 측정 — TTFT/TBT (완료 sink) + 3자원 idle (이론 대조용, S4d).

  TTFT = first_token_time − arrival_time
  TBT  = (completion_time − first_token_time) / (decoded_count − 1)     # decoded_count>1
  idle = idle_telemetry 의 gpu_a(GPU-A) / pim_a(PIM=A attention) / gpu_b(FFN=B)
         → spread = max−min = "가장 한가한 자원의 idle"(OPERATING_POINT §5 성공 기준).

프로토콜 (Phase-1 measure_steady 계승):
  1. warmup — REQUEST_ARRIVAL 전부 주입될 때까지 step (도착 transient 종료).
  2. reset — idle_telemetry.reset(now). cold-start transient 를 idle 에서 배제.
  3. measure — drain 까지 step. idle 누적 + 완료 sink 수집.
  4. TTFT/TBT 는 reset 이후 완료된 요청만(정상상태 윈도우). seed 요청은 TTFT 제외
     (실제 prefill 안 거침), TBT·idle 엔 포함(정상상태 디코드 발현원).

warm-start seed (--seed-decode-pool N, §2.6): t=0 에 prompt_len=0(이미 prefill 끝) 디코드
요청 N개 주입 → 정상상태 decode 풀 즉시 형성. cold-start 1M prefill 램프(수억 step)를
건너뛰어 실트레이스 정상상태 idle 을 현실적 비용에 측정. (정상 admission 경로가 prompt_len=0
요청을 decode-only mb 로 구성 — 별도 in-flight mb 수동 주입 불필요.)

실행: cd implementation && PYTHONIOENCODING=utf-8 python debug_phase2/measure_steady.py \
        --trace synthetic:200 --seed-decode-pool 200 --label s200 > debug_phase2/out_s200.txt
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puls_sched.event import Event, EventType
from puls_sched.request import Request
from puls_sched.run import Run

SEED_ID_BASE = 10_000_000        # seed 요청 id offset (트레이스 요청과 구분)


def pct(xs, p):
    """선형보간 percentile (numpy 비의존)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _fmt(xs):
    return (f"p50={pct(xs, 50):.1f} p90={pct(xs, 90):.1f} "
            f"max={max(xs) if xs else float('nan'):.1f} (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--config", default="puls_sched.config:default_dummy_config")
    ap.add_argument("--seed-decode-pool", type=int, default=0,
                    help="t=0 주입 decode 요청 수 (warm-start, §2.6)")
    ap.add_argument("--seed-ctx-mean", type=int, default=100_000)
    ap.add_argument("--seed-ctx-std", type=int, default=30_000)
    ap.add_argument("--seed-decode-remaining", type=int, default=200,
                    help="seed 요청 잔여 decode 토큰 수")
    ap.add_argument("--warmup-cap", type=int, default=20_000_000)
    ap.add_argument("--measure-cap", type=int, default=50_000_000)
    ns = ap.parse_args()

    run = Run.init(ns.config, ns.trace, f"debug_phase2/_scratch_{ns.label}")
    sched = run.scheduler
    tel = sched.admission.idle_telemetry

    # --- warm-start seed: prompt_len=0(이미 prefill) decode 요청 t=0 주입 ---
    if ns.seed_decode_pool > 0:
        rng = random.Random(sched.config.seed)
        for j in range(ns.seed_decode_pool):
            ctx = max(1_000, int(rng.gauss(ns.seed_ctx_mean, ns.seed_ctx_std)))
            req = Request(id=SEED_ID_BASE + j, prompt_len=0, kv_length=ctx,
                          arrival_time=0.0, max_tokens=ns.seed_decode_remaining)
            sched.queue.push(Event(timestamp=0.0, type=EventType.REQUEST_ARRIVAL,
                                   payload={"request": req}))

    def arrivals_pending():
        return any(ev.type == EventType.REQUEST_ARRIVAL for (_, _, ev) in sched.queue._heap)

    def drained():
        return (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
                and len(sched.in_flight_requests) == 0)

    # --- 1. warmup — 도착 전부 주입될 때까지 (도착 transient 종료) ---
    w = 0
    while w < ns.warmup_cap and not drained() and arrivals_pending():
        if not sched.step():
            break
        w += 1

    # --- 2. reset — cold-start transient 를 idle 에서 배제 ---
    reset_t = sched.clock.now
    tel.reset(reset_t)

    # --- 3. measure — drain 까지 ---
    m = 0
    while m < ns.measure_cap and not drained():
        if not sched.step():
            break
        m += 1

    # --- 4. TTFT/TBT (정상상태 = reset 이후 완료) ---
    steady = [r for r in sched.completed_requests
              if r.completion_time is not None and r.completion_time >= reset_t]
    ttft, tbt = [], []
    for r in steady:
        is_seed = r.id >= SEED_ID_BASE
        if r.first_token_time is not None and not is_seed:
            ttft.append(r.first_token_time - r.arrival_time)
        if r.first_token_time is not None and r.decoded_count > 1:
            tbt.append((r.completion_time - r.first_token_time) / (r.decoded_count - 1))

    g, p, b = (tel.gpu_idle_fraction(), tel.pim_idle_fraction(),
               tel.gpu_instance_b_idle_fraction())
    spread = max(g, p, b) - min(g, p, b)
    L = [
        f"=== steady measure: {ns.label} ===",
        f"trace={ns.trace} seed_pool={ns.seed_decode_pool} warmup_steps={w} measure_steps={m}",
        f"completed_total={len(sched.completed_requests)} steady(reset 이후)={len(steady)}",
        f"TTFT µs  {_fmt(ttft)}   (seed 제외, 실 prefill 요청만)",
        f"TBT  µs  {_fmt(tbt)}",
        f"idle  gpu_a={g:.4f} pim_a={p:.4f} gpu_b={b:.4f}   spread={spread:.4f}"
        f"  (= 가장 한가한 자원 idle; 동작점 명중 시 ~0)",
    ]
    out = "\n".join(L)
    print(out)
    Path(f"debug_phase2/steady_{ns.label}.txt").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
