from dataclasses import dataclass

from puls_sched.config import AdmissionConfig
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue


@dataclass(frozen=True)
class MicroBatchSpec:
    prefill_chunk_tokens: int
    decode_requests: tuple[Request, ...]
    kv_rows_total: int                                   # Impl-5 — Σ kv_length over decode_requests (signal flow to dispatcher, F5 활성화 path)
    kv_rows_lockstep: int                                # Impl-8 — max(kv_length) × num_decode_reqs (F5 ablation 위 lock-step penalty 산식)


@dataclass
class Admission:
    admission_cfg: AdmissionConfig
    request_queue: RequestQueue
    kv_accountant: KVAccountant
    idle_telemetry: IdleTelemetry

    def layer1(self, max_mb_kv_tokens: int | None = None) -> MicroBatchSpec | None:
        """Phase-2 S2 동작점 former — 디코더를 KV 합 동작점(§0.8)까지 admit + prefill 512 고정.

        밸런스가 정적 동작점(KV 합 25M + prefill 512)으로 확정되어(§0.8) **매 tick 동적
        측정·계산이 통째로 moot**(§2.5): t_proj·t_pim_fn·a_cycle·b_cycle·balance_* 인자/호출
        전부 제거. former 는 "디코더를 골라 담아 Σkv 를 동작점에 맞춤" 한 가지만 한다.

        종료 조건 (요청 *개수* 캡 없음 — §0.8 사용자 확정):
        - (i) Σkv ≥ `kv_operating_target_tokens` (동작점 25M, §0.8). 디코드는 쪼개 넣을 수
          없어(요청 KV 통째) 마지막 디코더가 목표를 살짝 넘겨 허용 밴드 [21.5M,29M] 안에 안착.
        - (ii) 전역 KV 부족(`can_admit`) — 총 60M aggregate 천장.
        - (iii) per-mb KV 예산 `max_mb_kv_tokens` (= 60M/2 = 30M, 2-슬롯 disjoint 분할, §0.8 A안).

        **N_dec(디코더 수)은 레버가 아니라 부산물.** 동작점은 (decode KV 25M, prefill 512)
        둘로 정의되고, t_B=FFN(N_dec+512)·t_GPU-A 의 projection 도 그 둘에서 자연 결정됨
        (N_dec = 25M ÷ ctx). 타깃 ctx ~100K → N_dec≈250 → 셋 다 ~101µs. seq 개수 캡은 KV 목표가
        먼저 멈춰 안 걸리는 중복 레버라 제거 — 채울 디코더를 인위로 거부하지 않는다.

        풀이 동작점을 못 채우면(짧은 요청만, 저volume) 작은 배치로 자연 수용 — PIM 유휴는
        고칠 대상이 아니라 물리적 정상(§0.9). prefill = `prefill_chunk_default`(512) 고정:
        REPORT prefill sweep 에서 1024+ 는 PREFILL_ATTN=O(prefill×ctx) 폭주로 균형 불가, 512 만 삼중 균형.
        """
        # Impl-9 — Head-of-line skip (production scheduler 정합, vLLM/Sarathi 영역).
        # FIFO break 대신 queue 전체 walk + fit 가능한 것만 admit. 큰 req 는 *상대 순서 유지*
        # 위 re-push → 다음 tick 위 capacity 회수 후 재시도.
        candidates: list[Request] = []
        while True:
            req = self.request_queue.pop_oldest()
            if req is None:
                break
            candidates.append(req)

        target = self.admission_cfg.kv_operating_target_tokens
        decode_reqs: list[Request] = []
        mb_kv = 0
        reached_target = False
        for req in candidates:
            # per-mb 예산: 빈 mb 는 첫 req 무조건 허용(단일 거대요청 starvation 방지) 후 예산 적용.
            fits_mb_kv = (
                max_mb_kv_tokens is None
                or not decode_reqs
                or mb_kv + req.kv_length <= max_mb_kv_tokens
            )
            if (not reached_target
                    and self.kv_accountant.can_admit(req) and fits_mb_kv):
                self.kv_accountant.admit(req)
                decode_reqs.append(req)
                mb_kv += req.kv_length
                if mb_kv >= target:
                    # 동작점 도달 — 남은 디코더는 다른 슬롯/다음 tick 의 몫.
                    reached_target = True
            else:
                # Defer — queue 의 그 자리 (상대 순서 유지) 위 re-push
                # (동작점 도달 / seq 상한 / KV 부족 / per-mb 예산 초과 중 하나)
                if not self.request_queue.push(req):
                    raise RuntimeError(
                        f"admission re-push failed for req {req.id} "
                        f"(RequestQueue capacity violation — defensive raise)"
                    )

        if not decode_reqs:
            return None

        # prefill 512 고정 (동적 chunk 사이징 삭제, §2.5). decode 조절 레버 없음(부산물).
        # N_dec(개수)도 부산물 — mfu_floor 클램프 삭제(§2.5, 동작점 N_dec≈250≫n_sat 이라 사문).
        prefill_chunk_tokens = self.admission_cfg.prefill_chunk_default
        kv_rows_total = sum(r.kv_length for r in decode_reqs)
        # Impl-8 — F5 ablation 위 lock-step penalty 입력. decode_reqs 비어 있으면 0.
        kv_rows_lockstep = max((r.kv_length for r in decode_reqs), default=0) * len(decode_reqs)

        return MicroBatchSpec(
            prefill_chunk_tokens=prefill_chunk_tokens,
            decode_requests=tuple(decode_reqs),
            kv_rows_total=kv_rows_total,
            kv_rows_lockstep=kv_rows_lockstep,
        )
