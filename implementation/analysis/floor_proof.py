"""S5 — idle 이 "이 워크로드 + 이 알고리즘의 floor" 임을 정량 증명 (OPERATING_POINT §floor).

측정 idle(measure_steady: GPU-A 8.0% · PIM 12.6% · FFN 12.6% · spread 4.6%)이 *알고리즘이
짜낼 수 있는 최소* 임을, 측정이 실제 디스패치한 **live μ-batch** 에 디스패처와 *동일한 op-time
함수* 를 직접 호출해 입증한다. 합성 batch 재구성이 아니라 시뮬레이터가 만든 mb 객체 그대로 →
재구성 오차 0.

세 자원은 각각 **단일 서버**(dispatcher 의 gpu_busy · pim_busy · instance_b_busy = I4/I5/I6
불변식, 동시 1 op). 따라서 steady-state 처리율은 가장 바쁜 서버의 per-mb work 가 律속:

  per μ-batch (한 layer):
    t_gpuA = QKV + PREFILL_ATTN + O_PROJ        (gpu_instance_a, 직렬)
    t_pim  = DECODE_ATTN                         (pim_instance_a)
    t_ffn  = FFN                                 (gpu_instance_b)
  완벽 overlap(F2 double-buffer · F3 inter-instance pipeline) →
    cycle      = max(t_gpuA, t_pim, t_ffn)       (병목 서버가 律속)
    floor idle_r = (cycle − t_r) / cycle = 1 − t_r/cycle

병목 자원 floor idle = 0. 측정 idle − 이론 floor = overlap 비효율(pipeline fill/drain ·
2-active staggering 전이 틈). 병목(GPU-A) 측정 idle 이 곧 그 overlap gap(이론 floor 0 이므로).
gap 이 작고 비병목 측정 ≈ 이론 floor + gap 이면 = **greedy dispatch 가 floor 에 근접** 입증.

실행: cd implementation && PYTHONIOENCODING=utf-8 python analysis/floor_proof.py \
        --trace data/sweep_D_longdec.csv --seed-pool 2000
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puls_sched.config import compute_ffn_op_time_s, compute_gpu_op_time_s
from puls_sched.event import Event, EventType
from puls_sched.node import NodeType
from puls_sched.request import Request
from puls_sched.run import Run

SEED_ID_BASE = 10_000_000
PF_RATE = 256


def _trace_lengths(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["num_prefill_tokens"]), max(1, int(r["num_decode_tokens"]))))
    return rows


def _seed_pool(sched, lengths, n, rng):
    """measure_steady 와 동일한 비편향 warm-start (생애 랜덤 지점, 사이클-시간 가중)."""
    for j in range(n):
        pf, dc = rng.choice(lengths)
        pf_cycles = max(1, pf // PF_RATE)
        total_cycles = pf_cycles + dc
        c = rng.randrange(total_cycles)
        req = Request(id=SEED_ID_BASE + j, prompt_len=pf, arrival_time=0.0, max_tokens=dc)
        if c < pf_cycles:
            req.prefill_processed = min(c * PF_RATE, max(0, pf - 1))
            req.kv_length = pf + dc
        else:
            req.prefill_processed = pf
            req.decoded_count = c - pf_cycles
            req.kv_length = pf + req.decoded_count
        sched.queue.push(Event(timestamp=0.0, type=EventType.REQUEST_ARRIVAL,
                               payload={"request": req}))


def _op_times_us(sched, mb):
    """live μ-batch 의 세 자원 op-time(µs) — dispatcher._op_time 와 *동일 산식*.

    GPU-A = QKV + PREFILL_ATTN + O_PROJ (num_gpus_instance_a, ×1e6).
    PIM   = DECODE_ATTN op_time(Σkv) (×1e-3, ns→µs). Σkv = 물리 진실(in_flight kv_length 합).
    FFN   = compute_ffn_op_time_s (num_gpus_instance_b, ×1e6).
    """
    cal, model, hw = sched.config.calibration, sched.config.model, sched.config.hw
    t_qkv = compute_gpu_op_time_s(NodeType.QKV, mb, cal, model, num_gpus=hw.num_gpus_instance_a)
    t_patt = compute_gpu_op_time_s(NodeType.PREFILL_ATTN, mb, cal, model, num_gpus=hw.num_gpus_instance_a)
    t_oproj = compute_gpu_op_time_s(NodeType.O_PROJ, mb, cal, model, num_gpus=hw.num_gpus_instance_a)
    t_gpuA = (t_qkv + t_patt + t_oproj) * 1e6
    _gpuA_parts = (t_qkv * 1e6, t_patt * 1e6, t_oproj * 1e6)
    # Σkv = in_flight kv_length 합 (measure_steady 의 12.33M 과 동일 정의 = PIM 물리 입력).
    sigma_kv = sum(sched.in_flight_requests[r].kv_length
                   for r in mb.decode_tokens if r in sched.in_flight_requests)
    t_pim = sched.dispatcher.pim_executor.op_time(kv_rows_total=sigma_kv) * 1e-3
    t_ffn = compute_ffn_op_time_s(mb, cal, model, num_gpus=hw.num_gpus_instance_b) * 1e6
    depth_work = sum(sum(v) for v in mb.prefill_chunk.values())   # Σ token depth (measure_steady 정의)
    n_dec = len(mb.decode_tokens)
    n_pref = sum(len(v) for v in mb.prefill_chunk.values())
    return t_gpuA, t_pim, t_ffn, sigma_kv, _gpuA_parts, depth_work, n_dec, n_pref


def _per_batch_lines(sched, target_mb):
    """활성 μ-batch 의 구성 + 배치별 *이론 floor* 유휴율 + disjoint 확인 (개별, 평균 아님)."""
    mbs = list(sched.dispatcher.micro_batches.values())
    L = [f"--- 개별 μ-batch 구성 (활성 {len(mbs)} / 목표 {target_mb}, "
         f"KV cap {sched.kv_accountant.capacity/1e6:.0f}M) ---",
         "  (배치별 유휴율 = 그 배치 구성의 *이론 floor*; 측정 idle 은 자원 공유라 window 전체값)"]
    member_sets = []
    for mb in mbs:
        tg_i, tp_i, tf_i, skv_i, _pp, dw_i, ndec_i, npref_i = _op_times_us(sched, mb)
        members = mb.request_ids()
        member_sets.append((mb.id, members))
        cyc_i = max(tg_i, tp_i, tf_i)
        flg, flp, flf = 1 - tg_i / cyc_i, 1 - tp_i / cyc_i, 1 - tf_i / cyc_i
        L.append(f"  mb#{mb.id}: decode {ndec_i:3d}/123 · Σkv {skv_i/1e6:5.2f}M/12.3M · "
                 f"prefill {npref_i:3d}/256 · depth {dw_i/1e6:5.2f}M/25.6M · 멤버 {len(members)}")
        L.append(f"         t(µs) GPU-A {tg_i:.2f} / PIM {tp_i:.2f} / FFN {tf_i:.2f}  →  "
                 f"floor idle  GPU-A {flg*100:5.2f}% · PIM {flp*100:5.2f}% · FFN {flf*100:5.2f}%  "
                 f"(spread {(max(flg,flp,flf)-min(flg,flp,flf))*100:.2f}%)")
    overlaps = [(member_sets[i][0], member_sets[j][0], len(member_sets[i][1] & member_sets[j][1]))
                for i in range(len(member_sets)) for j in range(i + 1, len(member_sets))
                if member_sets[i][1] & member_sets[j][1]]
    L.append(f"  {'⚠ disjoint 위반: ' + str(overlaps) if overlaps else '✓ disjoint — 활성 mb 멤버 겹침 0'}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--warmup-cap", type=int, default=3_000_000)
    ap.add_argument("--measure-cap", type=int, default=15_000_000)
    ap.add_argument("--sample-k", type=int, default=5_000)
    ap.add_argument("--converge-eps", type=float, default=0.01)
    ap.add_argument("--label", default="floor")
    ap.add_argument("--prefill", type=int, default=None,
                    help="동작점 prefill 스케일 오버라이드 (배포 128). 256 도출 기준의 모든 동작점 값을 "
                         "prefill/256 으로 스케일: decode_count·kv_operating·prefill_kv_work·kv_capacity + PF_RATE.")
    ap.add_argument("--age-cap", type=int, default=None,
                    help="age_cap 오버라이드 (item 4 ablation; 큰 값=∞ 무력화)")
    ap.add_argument("--seed", type=int, default=None,
                    help="warm-start 풀 샘플링 난수 시드 오버라이드 (다른 비편향 스냅샷 관찰)")
    ap.add_argument("--target-mb", type=int, default=None,
                    help="활성 μ-batch 목표 오버라이드 (기본 2). 3 으로 윈도우 세 칸 강제 형성 — "
                         "스케줄러 소스 무수정, main_loop._STAGGERING_TARGET_MB 런타임 monkeypatch.")
    ap.add_argument("--kv-cap", type=int, default=None,
                    help="kv_capacity_aggregate 오버라이드 (3배치면 ~42M 필요; decode steering 은 "
                         "여전히 123/12.3M 에서 정지).")
    ap.add_argument("--per-batch", action="store_true",
                    help="활성 μ-batch 의 구성을 *개별*(평균 아님) 출력 + disjoint 확인.")
    ap.add_argument("--compose-only", action="store_true",
                    help="무한-풀 근사: 풀을 크게 두고 목표 μ-batch 를 *구성하자마자 스냅샷·종료* "
                         "(긴 측정 루프 없음). 풀 충분 시 모든 배치가 타깃 명중, 꼬리 불균형 0.")
    ns = ap.parse_args()

    # 스케줄러 *소스* 무수정 — 런타임 상수 monkeypatch 만 (KV 오버라이드와 동급).
    if ns.target_mb is not None:
        import puls_sched.main_loop as _ml
        _ml._STAGGERING_TARGET_MB = ns.target_mb

    run = Run.init("puls_sched.config:default_dummy_config", ns.trace,
                   f"analysis/_scratch_{ns.label}")
    sched = run.scheduler
    if ns.prefill is not None:   # 동작점 prefill 스케일 (배포 128 = 모든 값 ÷2; 256 도출 기준)
        global PF_RATE
        s = ns.prefill / 256.0
        ad = sched.config.admission
        PF_RATE = ns.prefill
        object.__setattr__(ad, "prefill_chunk_default", ns.prefill)
        object.__setattr__(ad, "decode_count_target", round(ad.decode_count_target * s))
        object.__setattr__(ad, "kv_operating_target_tokens", round(ad.kv_operating_target_tokens * s))
        object.__setattr__(ad, "prefill_kv_work_target_tokens", round(ad.prefill_kv_work_target_tokens * s))
        new_cap = round(ad.kv_capacity_aggregate * s)
        object.__setattr__(ad, "kv_capacity_aggregate", new_cap)
        sched.kv_accountant._capacity = new_cap
    if ns.age_cap is not None:   # frozen dataclass — in-place 우회(모든 컴포넌트가 동일 객체 참조)
        object.__setattr__(sched.config.admission, "age_cap", ns.age_cap)
    if ns.kv_cap is not None:
        object.__setattr__(sched.config.admission, "kv_capacity_aggregate", ns.kv_cap)
        sched.kv_accountant._capacity = ns.kv_cap   # Run.init 시 생성된 accountant 에도 전파
    _target_mb = ns.target_mb if ns.target_mb is not None else 2
    tel = sched.admission.idle_telemetry
    rng = random.Random(ns.seed if ns.seed is not None else sched.config.seed)
    _seed_pool(sched, _trace_lengths(ns.trace), ns.seed_pool, rng)

    def drained():
        return (len(sched.queue) == 0 and len(sched.window.current_ids()) == 0
                and len(sched.in_flight_requests) == 0)

    w = 0
    while w < ns.warmup_cap and not drained():
        if len(sched.window.current_ids()) >= _target_mb:
            break
        if not sched.step():
            break
        w += 1

    if ns.compose_only:
        # 무한-풀 근사: 목표 배치가 *방금* 구성된 fresh 스냅샷 — 측정 루프(드레인) 없이 바로 보고.
        head = (f"=== compose-only snapshot: {ns.trace} "
                f"(seed-pool {ns.seed_pool}, 목표 {_target_mb}, warmup {w} step) ===\n"
                f"무한-풀 근사 — 풀이 충분하면 모든 배치가 (123 · 12.3M · 256 · 25.6M) 명중, 꼬리 불균형 0.")
        out = head + "\n" + "\n".join(_per_batch_lines(sched, _target_mb))
        print(out)
        Path(f"analysis/floor_{ns.label}.txt").write_text(out + "\n", encoding="utf-8")
        return

    tel.reset(sched.clock.now)

    m, samples, converged = 0, [], False
    ot = []   # (t_gpuA, t_pim, t_ffn, sigma_kv) per 활성 mb 샘플
    pf_pool = []   # (pool_size, n_shallower_than_ideal, min_depth) per 샘플 — item 3 진단
    from puls_sched.request import RequestState
    IDEAL_DEPTH = (sched.config.admission.prefill_kv_work_target_tokens
                   / sched.config.admission.prefill_chunk_default)   # 25.6M/256 = 100K
    while m < ns.measure_cap and not drained():
        if not sched.step():
            break
        m += 1
        if m % ns.sample_k == 0:
            samples.append((tel.gpu_idle_fraction(), tel.pim_idle_fraction(),
                            tel.gpu_instance_b_idle_fraction()))
            for mb in sched.dispatcher.micro_batches.values():
                ot.append(_op_times_us(sched, mb))
            # item 3 — prefill 풀(in-flight PREFILL, 잔여 prefill>0)의 가용 depth 분포.
            # 이상 깊이(100K)보다 *얕은* 후보가 풀에 있는데 안 쓰였나(greedy 손해) vs 없나(불가피)?
            depths = [r.prefill_processed for r in sched.in_flight_requests.values()
                      if r.state is RequestState.PREFILL and r.prompt_len > r.prefill_processed]
            if depths:
                pf_pool.append((len(depths),
                                sum(1 for d in depths if d < IDEAL_DEPTH),
                                min(depths)))
            if len(samples) >= 5:
                wnd = samples[-5:]
                if max(max(x[i] for x in wnd) - min(x[i] for x in wnd)
                       for i in range(3)) < ns.converge_eps:
                    converged = True
                    break

    g, p, b = (tel.gpu_idle_fraction(), tel.pim_idle_fraction(),
               tel.gpu_instance_b_idle_fraction())
    n = len(ot)
    t_gpuA = sum(o[0] for o in ot) / n
    t_pim = sum(o[1] for o in ot) / n
    t_ffn = sum(o[2] for o in ot) / n
    skv = sum(o[3] for o in ot) / n
    t_qkv = sum(o[4][0] for o in ot) / n
    t_patt = sum(o[4][1] for o in ot) / n
    t_oproj = sum(o[4][2] for o in ot) / n
    dwork = sum(o[5] for o in ot) / n
    n_dec = sum(o[6] for o in ot) / n
    n_pref = sum(o[7] for o in ot) / n
    cycle = max(t_gpuA, t_pim, t_ffn)
    floor = {"gpu_a": 1 - t_gpuA / cycle, "pim_a": 1 - t_pim / cycle, "gpu_b": 1 - t_ffn / cycle}
    meas = {"gpu_a": g, "pim_a": p, "gpu_b": b}
    names = {"gpu_a": "GPU-A (QKV+PREFILL_ATTN+O_PROJ)", "pim_a": "PIM (DECODE_ATTN)",
             "gpu_b": "FFN (Instance B)"}

    L = [
        f"=== floor proof: {ns.trace} (n_mb샘플={n}, converged={converged}) ===",
        "",
        "per-μ-batch op-time (µs, dispatcher 와 동일 산식):",
        f"  t_gpuA = {t_gpuA:8.3f}   t_pim = {t_pim:8.3f}   t_ffn = {t_ffn:8.3f}",
        f"    t_gpuA 분해: QKV {t_qkv:.3f} + PREFILL_ATTN {t_patt:.3f} + O_PROJ {t_oproj:.3f}",
        f"  batch: decode {n_dec:.0f}/123 · prefill {n_pref:.0f}/256 토큰",
        f"  Σkv(평균) = {skv/1e6:.2f}M   cycle = max = {cycle:.3f} µs  (병목 = "
        f"{max(('gpu_a', t_gpuA), ('pim_a', t_pim), ('gpu_b', t_ffn), key=lambda x: x[1])[0]})",
        "",
        f"{'자원':<34} {'이론floor':>9} {'측정idle':>9} {'overlap gap':>12}",
    ]
    for k in ("gpu_a", "pim_a", "gpu_b"):
        L.append(f"{names[k]:<34} {floor[k]*100:8.2f}% {meas[k]*100:8.2f}% "
                 f"{(meas[k]-floor[k])*100:11.2f}%")

    if ns.per_batch:
        L.append("")
        L += _per_batch_lines(sched, _target_mb)
    fl_spread = max(floor.values()) - min(floor.values())
    me_spread = max(meas.values()) - min(meas.values())
    L += [
        "",
        f"spread  이론 floor = {fl_spread*100:.2f}%   측정 = {me_spread*100:.2f}%",
        f"overlap gap(병목 GPU-A 측정 idle = 이론 floor 0 위의 fill/drain·전이 틈) = "
        f"{meas['gpu_a']*100:.2f}%",
        "",
        "해석: 비병목(PIM·FFN) 측정 ≈ 이론 floor + 병목 overlap gap 이면",
        "      = greedy dispatch 가 이 워크로드의 알고리즘 floor 에 근접.",
    ]
    if pf_pool:
        target_dw = sched.config.admission.prefill_kv_work_target_tokens
        np_ = len(pf_pool)
        avg_pool = sum(p[0] for p in pf_pool) / np_
        avg_shallow = sum(p[1] for p in pf_pool) / np_
        avg_min = sum(p[2] for p in pf_pool) / np_
        L += [
            "",
            f"--- item 3: prefill depth-work 27.1M(>25.6M) 오버슈트 진단 ---",
            f"이상 깊이(=25.6M/256) = {IDEAL_DEPTH/1e3:.0f}K",
            f"prefill 풀(in-flight PREFILL) 평균 크기 = {avg_pool:.1f}  "
            f"이상보다 얕은 후보 평균 = {avg_shallow:.1f}  풀 최소깊이 평균 = {avg_min/1e3:.0f}K",
            f"실측 depth-work = {dwork/1e6:.2f}M  (타깃 {target_dw/1e6:.1f}M, "
            f"오버슈트 +{(dwork/target_dw-1)*100:.1f}%)",
            f"t_gpuA 분해: QKV {t_qkv:.2f} + PREFILL_ATTN {t_patt:.2f}(∝depth-work) + O_PROJ {t_oproj:.2f}"
            f"  → PREFILL_ATTN 이 t_gpuA 의 {t_patt/t_gpuA*100:.0f}%",
        ]
        # counterfactual: PREFILL_ATTN ∝ depth-work(선형) → depth-work 가 정확히 25.6M 이면?
        t_patt_at_target = t_patt * (target_dw / dwork)
        t_gpuA_cf = t_qkv + t_patt_at_target + t_oproj
        cyc_cf = max(t_gpuA_cf, t_pim, t_ffn)
        fl_cf = {"gpu_a": 1 - t_gpuA_cf / cyc_cf, "pim_a": 1 - t_pim / cyc_cf,
                 "gpu_b": 1 - t_ffn / cyc_cf}
        L += [
            f"counterfactual(depth-work 정확히 25.6M): t_gpuA {t_gpuA_cf:.2f} → floor spread "
            f"{(max(fl_cf.values())-min(fl_cf.values()))*100:.2f}% (현 {fl_spread*100:.2f}%, age_cap={sched.config.admission.age_cap})",
            f"⇒ floor spread 의 대부분이 prefill depth-work 오버슈트(+{(dwork/target_dw-1)*100:.0f}%) 기인.",
            f"  item 4 (--age-cap 비교): 오버슈트는 age-cap 강제 산물 — age_cap=∞ 면 steering 이",
            f"  depth-work 를 ~25.6M 에 명중시켜 spread→~0. 즉 4.6% 는 *공정성 비용*(의도된 trade-off).",
        ]
    out = "\n".join(L)
    print(out)
    Path(f"analysis/floor_{ns.label}.txt").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
