from dataclasses import dataclass

from puls_sched.config import AdmissionConfig
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue


@dataclass
class Admission:
    admission_cfg: AdmissionConfig
    request_queue: RequestQueue
    kv_accountant: KVAccountant
    idle_telemetry: IdleTelemetry

    def steer_decode_set(self, candidates: list[Request]) -> list[Request]:
        """Phase-2 former-v2 (풀 모델) — decode 풀에서 로컬 그리디 steering+age-cap 으로 decode-set 선택.

        한 μ-batch 의 decode-set 을 *인플라이트 DECODE 풀*(candidates)에서 두 타깃에 맞춰 고른다
        (OPERATING_POINT §3): 개수 123(decode_count_target) AND Σkv 12.3M(kv_operating_target).
        이건 prefill 구성(256 토큰, main_loop)과 *완전히 별개* — decode 축만.

        ```
        n=0, S=0
        while n < target_count(123) and S < target_kv(12.3M) and pool:
          if (wait ≥ age_cap 인 디코더 있음): 가장 오래 기다린 것 선택   # 공정성(강제)
          else: ideal=(target_kv−S)/(target_count−n) 에 가장 가까운 디코더 선택  # steering
        미선택 디코더 wait += 1   # 다음 μ-batch 후보(풀에 잔류)
        ```

        - **KV 게이팅 없음**: 후보는 *이미 풀 진입 시 KV admit 된 인플라이트 디코더*(admission
          이 풀 보충 시 처리). 여기선 *선택*만 — admit/큐 조작 안 함.
        - **age-cap**: pure steering 의 off-size starvation 해소(wait≥cap 강제) → starvation 0,
          대기 ≤age_cap+1. **길이분산 무관**: avg 안 봄, 두 타깃만 맞춤(검증 proto_steering_fair).
        - 선택분 wait=0 리셋, 미선택분 wait+=1(다음 μ-batch 후보).
        """
        target_kv = self.admission_cfg.kv_operating_target_tokens
        target_count = self.admission_cfg.decode_count_target
        age_cap = self.admission_cfg.age_cap

        pool = list(candidates)
        selected: list[Request] = []
        S = 0
        n = 0
        while n < target_count and S < target_kv and pool:
            aged = [r for r in pool if r.wait >= age_cap]
            if aged:
                pick = max(aged, key=lambda r: r.wait)        # 가장 오래 기다린 디코더 강제
            else:
                ideal = (target_kv - S) / (target_count - n)
                pick = min(pool, key=lambda r: abs(r.kv_length - ideal))
            pool.remove(pick)
            selected.append(pick)
            S += pick.kv_length
            n += 1

        for r in selected:
            r.wait = 0
        for r in pool:
            r.wait += 1
        return selected
