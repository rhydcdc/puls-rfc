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
