"""STEP 5 엄밀판 idle 측정 — 워밍업 → reset → 정상상태 수렴까지 (완주 안 함).

완주 불필요: 정상상태(포화 backlog) 유휴율이 수렴하면 정지. 완주 막판 ramp-down
꼬리(도착 끊겨 한쪽 자원 유휴) 오염 회피 + 비용 절감. before/after·스윕에 **동일
프로토콜** 적용 (절대값보다 delta 가 결론).

프로토콜:
1. 워밍업 — 이벤트 큐에 REQUEST_ARRIVAL 이 남아있는 한 step (전 도착 주입 = 도착
   transient 종료 = backlog 정상상태 진입). 단위(클록) 추측 없음. + mb_count>0.
2. reset — idle_telemetry.reset(clock.now). 콜드스타트 transient 측정 배제.
3. 측정 — SAMPLE_K step 마다 누적 idle 3-key 샘플. 최근 WINDOW_W 샘플의 슬롯별
   변동폭(max-min) 이 모두 EPS 미만이면 수렴 → 정지. 안전상한 step 동반.

인자: --trace (필수) --label --batch --theta-high --theta-low --no-join --config
공통 수렴 파라미터(전 측정 동일): 아래 상수.
"""
import argparse
import dataclasses
import sys

sys.path.insert(0, "debug_phase1")
from puls_sched.event import EventType
from puls_sched.request import RequestState
from puls_sched.run import Run

SAMPLE_K = 20_000        # 샘플 간격 (step)
WINDOW_W = 5             # 수렴 판정 윈도우 (샘플 수) → 100K step 안정 구간
EPS = 0.005             # 수렴 임계 (슬롯별 변동폭)
WARMUP_CAP = 5_000_000   # 워밍업 안전상한
MEASURE_CAP = 15_000_000  # 측정 안전상한


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--theta-high", type=float, default=None)
    ap.add_argument("--theta-low", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=None,
                    help="prefill_chunk_default override (토큰 버짓). prefill 가속 → decode 구간 도달")
    ap.add_argument("--no-join", action="store_true")
    ap.add_argument("--warmup-decode-frac", type=float, default=0.5,
                    help="워밍업 종료 = in_flight 의 decode 비율 이 값 도달 시. 높이면 더 순수한 "
                         "decode 구간 측정(prefill flood 배제). burst 트레이스에서 PIM-bound 보려면 0.9")
    ap.add_argument("--config", default="puls_sched.config:default_dummy_config")
    ns = ap.parse_args()

    run = Run.init(ns.config, ns.trace, f"debug_phase1/_scratch_s_{ns.label}")
    sched = run.scheduler

    # --- config override (sweep) ---
    repl = {}
    if ns.batch is not None:
        repl["max_batch_size"] = ns.batch
    if ns.theta_high is not None:
        repl["idle_theta_high"] = ns.theta_high
    if ns.theta_low is not None:
        repl["idle_theta_low"] = ns.theta_low
    if ns.chunk is not None:
        repl["prefill_chunk_default"] = ns.chunk
    if repl:
        new_adm = dataclasses.replace(sched.config.admission, **repl)
        sched.config = dataclasses.replace(sched.config, admission=new_adm)
        sched.admission.admission_cfg = new_adm

    # --- before (합류 비활성) — _try_join 무력화 ---
    if ns.no_join:
        sched._try_join = lambda active_req_ids: set()

    mb_count = 0
    orig_register = sched.dispatcher.register
    def traced(mb):
        nonlocal mb_count
        mb_count += 1
        return orig_register(mb)
    sched.dispatcher.register = traced

    tel = sched.admission.idle_telemetry

    def drained():
        return (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
                and len(sched.in_flight_requests) == 0)

    def arrivals_pending():
        return any(ev.type == EventType.REQUEST_ARRIVAL for (_, _, ev) in sched.queue._heap)

    max_window = 0

    DECODE_WARMUP_FRACTION = ns.warmup_decode_frac  # in_flight decode 비율 이만큼 → 해당 구간 측정

    def decode_fraction():
        infl = sched.in_flight_requests
        if not infl:
            return 0.0
        d = sum(1 for r in infl.values() if r.state == RequestState.DECODE)
        return d / len(infl)

    # --- 1. 워밍업 — 도착 주입 완료 + 배치가 decode-bound 진입(prefill 통과)까지 ---
    # (도착 transient + 초기 prefill 구간 둘 다 측정에서 배제. 합류 효과는 decode-bound
    #  정상상태에서 backlog prefill 로 GPU 채움이라, 그 regime 에 도달해야 함.)
    w_steps = 0
    while w_steps < WARMUP_CAP:
        if drained():
            break
        if not arrivals_pending() and mb_count > 0 and decode_fraction() >= DECODE_WARMUP_FRACTION:
            break
        if not sched.step():
            break
        max_window = max(max_window, len(sched.window.current_ids()))
        w_steps += 1

    # --- 2. reset ---
    tel.reset(sched.clock.now)

    # --- 3. 정상상태 측정 → 수렴까지 ---
    samples = []
    m_steps = 0
    converged = False
    drained_flag = False
    while m_steps < MEASURE_CAP:
        if drained():
            drained_flag = True
            break
        if not sched.step():
            drained_flag = True
            break
        max_window = max(max_window, len(sched.window.current_ids()))
        m_steps += 1
        if m_steps % SAMPLE_K == 0:
            samples.append((
                tel.gpu_idle_fraction(),
                tel.pim_idle_fraction(),
                tel.gpu_instance_b_idle_fraction(),
            ))
            if len(samples) >= WINDOW_W:
                wnd = samples[-WINDOW_W:]
                rng = max(
                    max(x[i] for x in wnd) - min(x[i] for x in wnd)
                    for i in range(3)
                )
                if rng < EPS:
                    converged = True
                    break

    g, p, b = (
        tel.gpu_idle_fraction(),
        tel.pim_idle_fraction(),
        tel.gpu_instance_b_idle_fraction(),
    )
    th = ns.theta_high if ns.theta_high is not None else "default"
    tl = ns.theta_low if ns.theta_low is not None else "default"
    L = [
        f"=== steady idle: {ns.label} ===",
        f"trace={ns.trace}",
        f"join={'OFF' if ns.no_join else 'ON'} batch={ns.batch or 'default'} "
        f"chunk={ns.chunk or 'default'} theta_high={th} theta_low={tl}",
        f"warmup_steps={w_steps} measure_steps={m_steps} "
        f"converged={converged} drained={drained_flag}",
        f"mb_count={mb_count} max_window={max_window}/{sched.window.capacity} "
        f"samples={len(samples)}",
        f"idle gpu_a={g:.4f} pim_a={p:.4f} gpu_b={b:.4f}",
    ]
    out = "\n".join(L)
    print(out)
    with open(f"debug_phase1/steady_{ns.label}.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
