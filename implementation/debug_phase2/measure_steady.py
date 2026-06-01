"""Phase-2 정상상태 측정 — 3자원 idle 수렴 (idle↔이론 검증, S4d) + TTFT/TBT.

**검증 철학 (사용자 확정 2026-06-02):** idle≈0 을 *맞추려 튜닝하지 않는다.* 현실적인
거대·다양한-길이 워크로드를 *고정*해두고 스케줄러를 돌려, idle 이 자연히 수렴하는 값을
*그대로* 보고한다. 알고리즘이 옳다면 idle≈0 은 튜닝 없이 발현(F2/F3). non-zero 로 수렴하면
그게 결과(실제 불균형). "0 나올 때까지"가 아니라 "수렴하면 멈추고 값 보고".

  idle = gpu_a(GPU-A: proj+prefill-attn) / pim_a(PIM: decode-attn) / gpu_b(FFN=B)
  spread = max−min = "가장 한가한 자원의 idle" (OPERATING_POINT §5 성공 기준; 명중 시 ~0).
  TTFT = first_token−arrival, TBT = (completion−first_token)/(decoded−1).

**대표 warm-start (§2.6, 비편향):** cold-start 램프(long-ctx prefill 을 depth 0 부터 = 수억 step)
회피. 워크로드 트레이스 *자체* 의 (prefill,decode) 길이를 샘플하고, 각 요청을 **생애 랜덤
지점**(work-weighted: t∈[0,prefill+decode) → t<prefill 이면 그 depth 의 mid-prefill, 아니면
(t−prefill) 토큰 decode 진행)에 둔다 = 실서버 정상상태 스냅샷. 길이·생애 분포가 모두
워크로드에서 나오고 *고정*이라 체리피킹 아님 — 램프만 건너뛸 뿐 idle 값은 안 건드림.

실행: cd implementation && PYTHONIOENCODING=utf-8 python debug_phase2/measure_steady.py \
        --trace data/longctx_longbench_lambda_3_40.csv --seed-pool 2000 --label lb > debug_phase2/out_lb.txt
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puls_sched.event import Event, EventType
from puls_sched.request import Request
from puls_sched.run import Run
from puls_sched.trace import TraceReplayer

SEED_ID_BASE = 10_000_000


def pct(xs, p):
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


def _trace_lengths(path):
    """워크로드의 (prefill, decode) 길이 목록 — seed 분포 = 워크로드 분포(비편향)."""
    if path.startswith("synthetic:"):
        reqs = list(TraceReplayer.synthesize(n=int(path.split(":")[1]), seed=42).replay())
        return [(r.prompt_len, max(1, r.max_tokens)) for r in reqs]
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["num_prefill_tokens"]), max(1, int(r["num_decode_tokens"]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="워크로드 (seed 길이분포 + 도착 스트림)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--config", default="puls_sched.config:default_dummy_config")
    ap.add_argument("--seed-pool", type=int, default=2000,
                    help="대표 warm-start 요청 수 (생애 랜덤 지점, 워크로드 분포 샘플)")
    ap.add_argument("--warmup-cap", type=int, default=3_000_000)
    ap.add_argument("--measure-cap", type=int, default=15_000_000)
    ap.add_argument("--sample-k", type=int, default=5_000, help="idle 샘플 간격(step)")
    ap.add_argument("--converge-eps", type=float, default=0.01, help="수렴 임계(슬롯 변동폭)")
    ns = ap.parse_args()

    run = Run.init(ns.config, ns.trace, f"debug_phase2/_scratch_{ns.label}")
    sched = run.scheduler
    tel = sched.admission.idle_telemetry
    rng = random.Random(sched.config.seed)
    lengths = _trace_lengths(ns.trace)

    # --- 대표 warm-start seed (비편향): 생애 랜덤 지점 스냅샷 ---
    # ★ 생애 지점은 *사이클-시간*으로 가중(점유율 ∝ 머문 시간) — prefill≈pf/256 사이클,
    # decode≈dc 사이클. (토큰-work 가중은 pf≫dc 라 디코더를 거의 0 으로 만드는 버그.)
    PF_RATE = 256       # prefill ~256 토큰/사이클(상한; 공유 시 느려지나 점유 근사엔 충분)
    for j in range(ns.seed_pool):
        pf, dc = rng.choice(lengths)
        pf_cycles = max(1, pf // PF_RATE)
        total_cycles = pf_cycles + dc
        c = rng.randrange(total_cycles)
        req = Request(id=SEED_ID_BASE + j, prompt_len=pf, arrival_time=0.0, max_tokens=dc)
        if c < pf_cycles:                       # mid-prefill: c 사이클치 prefill 진행
            req.prefill_processed = min(c * PF_RATE, max(0, pf - 1))
            req.kv_length = pf + dc             # 전체 reservation (KV 회계)
        else:                                   # decoding: (c−pf_cycles) 토큰 진행
            decoded = c - pf_cycles
            req.prefill_processed = pf
            req.decoded_count = decoded
            req.kv_length = pf + decoded        # 현재 KV (PIM decode-attn 입력)
        sched.queue.push(Event(timestamp=0.0, type=EventType.REQUEST_ARRIVAL,
                               payload={"request": req}))

    def drained():
        return (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
                and len(sched.in_flight_requests) == 0)

    # --- 1. warmup — 활성 μ-batch 목표(2) 채워질 때까지(파이프라인 가동) 또는 cap ---
    # (_fill_window 는 staggering 목표 2 까지만 채움 — capacity(3) 가 아님. seed 가 풀을
    #  즉시 데우므로 첫 ADMISSION_TICK 후 곧 2 도달 → 짧은 warmup.)
    w = 0
    while w < ns.warmup_cap and not drained():
        if len(sched.window.current_ids()) >= 2:
            break
        if not sched.step():
            break
        w += 1

    # --- 2. reset — cold-start transient 를 idle 에서 배제 ---
    reset_t = sched.clock.now
    tel.reset(reset_t)

    # --- 3. measure — idle 수렴까지(샘플) 또는 cap/drain. *수렴값을 그대로 보고* ---
    # + 배치 구성 진단: 활성 mb 의 (decode 개수, Σkv, prefill 토큰, depth-work) 가 타깃
    #   (123 / 12.3M / 256 / 25.6M) 에 실제 맞는지 — idle 해석의 근거.
    m, samples, converged = 0, [], False
    comp = []   # (decode_count, decode_kv, prefill_tokens, depth_work) per 활성 mb 샘플
    while m < ns.measure_cap and not drained():
        if not sched.step():
            break
        m += 1
        if m % ns.sample_k == 0:
            samples.append((tel.gpu_idle_fraction(), tel.pim_idle_fraction(),
                            tel.gpu_instance_b_idle_fraction()))
            for mb in sched.dispatcher.micro_batches.values():
                dkv = sum(sched.in_flight_requests[r].kv_length
                          for r in mb.decode_tokens if r in sched.in_flight_requests)
                ptok = sum(len(v) for v in mb.prefill_chunk.values())
                dwork = sum(sum(v) for v in mb.prefill_chunk.values())
                pref_reqs = len(mb.prefill_chunk)           # cohort 의 prefill 요청 수
                comp.append((len(mb.decode_tokens), dkv, ptok, dwork, pref_reqs))
            if len(samples) >= 5:
                wnd = samples[-5:]
                if max(max(x[i] for x in wnd) - min(x[i] for x in wnd) for i in range(3)) < ns.converge_eps:
                    converged = True
                    break

    steady = [r for r in sched.completed_requests
              if r.completion_time is not None and r.completion_time >= reset_t]
    ttft, tbt = [], []
    for r in steady:
        if r.first_token_time is not None and r.id < SEED_ID_BASE:   # seed 는 TTFT 제외
            ttft.append(r.first_token_time - r.arrival_time)
        if r.first_token_time is not None and r.decoded_count > 1:
            tbt.append((r.completion_time - r.first_token_time) / (r.decoded_count - 1))

    g, p, b = (tel.gpu_idle_fraction(), tel.pim_idle_fraction(),
               tel.gpu_instance_b_idle_fraction())
    L = [
        f"=== steady measure: {ns.label} ===",
        f"trace={ns.trace} seed_pool={ns.seed_pool} warmup_steps={w} measure_steps={m} converged={converged}",
        f"completed_total={len(sched.completed_requests)} steady(reset 이후)={len(steady)}",
        f"TTFT µs  {_fmt(ttft)}   (seed 제외, 실 도착 요청만)",
        f"TBT  µs  {_fmt(tbt)}",
        f"idle  gpu_a={g:.4f} pim_a={p:.4f} gpu_b={b:.4f}   spread={max(g,p,b)-min(g,p,b):.4f}"
        f"  (가장 한가한 자원 idle; 동작점 명중 시 ~0)",
    ]
    if comp:
        n = len(comp)
        dc = sum(c[0] for c in comp) / n
        dkv = sum(c[1] for c in comp) / n
        pt = sum(c[2] for c in comp) / n
        dw = sum(c[3] for c in comp) / n
        pr = sum(c[4] for c in comp) / n
        L.append(
            f"batch(활성 mb avg, 타깃대비): decode개수={dc:.0f}/123  Σkv={dkv/1e6:.2f}M/12.3M  "
            f"prefill토큰={pt:.0f}/256  depth-work={dw/1e6:.2f}M/25.6M  (n_mb샘플={n})")
        L.append(
            f"  cohort 구성: decode req {dc:.0f} + prefill req {pr:.0f} = {dc+pr:.0f} (layer1 admit 단위)")
    out = "\n".join(L)
    print(out)
    Path(f"debug_phase2/steady_{ns.label}.txt").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
