"""Debug 전용 config — KV 캐파만 작게(직렬 검증 가속용). 소스 무수정.

직렬 여부(첫 mb 완료 후 2번째 mb 생성)는 캐파 크기와 무관한 구조적 성질이므로,
캐파를 작게 줄여 작은 트레이스로 빠르게 overflow → 첫 mb 완료 → 2번째 mb 관측.
"""
import dataclasses

from puls_sched.config import default_dummy_config


def small_cap_config():
    cfg = default_dummy_config()
    adm = dataclasses.replace(cfg.admission, kv_capacity_aggregate=200_000)
    return dataclasses.replace(cfg, admission=adm)


def small_cap_seqlimit_config():
    """STEP 1 검증용 — KV 캐파 200K + seq 상한 2.

    same 트레이스(trace_serial_tiny)를 small_cap_config(seq 무제한)와 비교:
    - seq 무제한: 첫 mb 가 3개(180K) 점유 → 직렬 (max window=1)
    - seq=2: 첫 mb 2개(120K), 남은 80K 로 mb1 1개(60K) 동시 생성 → max window>1
    변수 하나(seq 상한)만 바꿔 mb 다중화 인과 분리.
    """
    cfg = default_dummy_config()
    adm = dataclasses.replace(
        cfg.admission, kv_capacity_aggregate=200_000, max_batch_size=2,
    )
    return dataclasses.replace(cfg, admission=adm)
