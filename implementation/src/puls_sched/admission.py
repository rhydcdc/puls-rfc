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
        """Phase-2 former-v2 — 로컬 그리디 steering + age-cap 으로 decode 배치 구성.

        동작점은 *두 타깃 동시 만족*으로 정의된다(OPERATING_POINT §3): decode **개수 123**
        (FFN batch 균형) AND decode **Σkv 12.3M**(PIM 균형). 디코드는 쪼개 넣을 수 없어
        (요청 KV 통째) FIFO 로 Σkv 만 채우면 *개수* 를 놓쳐 off-avg 풀에서 어긋난다
        (검증: spread 22~30%). 그래서 매 step **"다음에 필요한 길이"** 를 산출해 그에 가장
        가까운 디코더를 고른다(steering) → 두 타깃 동시 자기보정 수렴(검증 spread~1%).

        ```
        n=0, S=0
        while n < target_count(123) and S < target_kv(12.3M) and 후보 있음:
          if (wait ≥ age_cap 인 후보 있음): 가장 오래된 그것 admit   # 공정성(강제)
          else: ideal=(target_kv−S)/(target_count−n) 에 가장 가까운 디코더 admit  # steering
        나머지 후보 wait += 1   # 다음 batch 후보로 re-push
        ```

        - **age-cap**: pure steering 은 ideal-크기만 cherry-pick → off-size starvation. wait ≥
          age_cap(2) 강제 포함 → starvation 0(검증: 서빙분포=arrival분포) + 대기 ≤age_cap+1.
          강제된 off-size 도 steering 이 다음 step 에서 보정(ideal 이동)해 배치 균형 유지.
        - **closest-to-ideal 이 오버슈트 자체 방지** → 상한 밴드 가드 불필요(밴드는 진단용).
        - **길이분산 무관**: avg 안 봄, 두 타깃만 맞춤 → 어떤 길이분포든 거대 풀서 조합으로 동작.

        용량 게이팅(steering 과 직교): 전역 KV(`can_admit`, 총 30M) + per-mb 예산
        (`max_mb_kv_tokens`=30M/2=15M, 2-슬롯 disjoint). 매 step *수용 가능한* 후보 중에서만
        고른다 — 빈 batch 의 첫 후보는 per-mb 예산 초과(단일 거대요청)도 허용(starvation 방지).
        풀이 타깃을 못 채우면(짧은 요청만) 작은 배치로 자연 수용 — PIM 유휴는 물리적 정상.
        prefill 토큰 분배(256, depth-합 25.6M steering)는 main_loop._populate_mb_phases 가 담당.
        """
        # 후보 풀 = request_queue 전체 drain (arrival 순서 보존 — age-cap 의 "오래된 것" 기준).
        candidates: list[Request] = []
        while True:
            req = self.request_queue.pop_oldest()
            if req is None:
                break
            candidates.append(req)

        target_kv = self.admission_cfg.kv_operating_target_tokens
        target_count = self.admission_cfg.decode_count_target
        age_cap = self.admission_cfg.age_cap

        decode_reqs: list[Request] = []
        mb_kv = 0
        n = 0

        # 용량 수용 가능 여부 (steering 과 직교). 첫 후보는 per-mb 예산 면제(단일 거대요청
        # starvation 방지) — 호출 시점의 mb_kv·decode_reqs 를 읽는다.
        def _fits(r: Request) -> bool:
            if not self.kv_accountant.can_admit(r):
                return False
            if max_mb_kv_tokens is None or not decode_reqs:
                return True
            return mb_kv + r.kv_length <= max_mb_kv_tokens

        while n < target_count and mb_kv < target_kv and candidates:
            fitting = [r for r in candidates if _fits(r)]
            if not fitting:
                break
            aged = [r for r in fitting if r.wait >= age_cap]
            if aged:
                pick = aged[0]                          # arrival 순서 보존 → 가장 오래된 aged
            else:
                ideal = (target_kv - mb_kv) / (target_count - n)
                pick = min(fitting, key=lambda r: abs(r.kv_length - ideal))
            self.kv_accountant.admit(pick)
            candidates.remove(pick)
            decode_reqs.append(pick)
            mb_kv += pick.kv_length
            n += 1

        # 미선택 후보 → wait += 1 후 re-push (상대 순서 = arrival 순서 보존).
        for req in candidates:
            req.wait += 1
            if not self.request_queue.push(req):
                raise RuntimeError(
                    f"admission re-push failed for req {req.id} "
                    f"(RequestQueue capacity violation — defensive raise)"
                )

        if not decode_reqs:
            return None

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
