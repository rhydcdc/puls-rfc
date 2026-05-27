from puls_sched.config import AdmissionConfig


def lookup_width(admission_cfg: AdmissionConfig, ctx_tokens: int) -> float:
    if ctx_tokens <= admission_cfg.ctx_tier_short_max:
        return admission_cfg.deadband_width["short"]
    if ctx_tokens <= admission_cfg.ctx_tier_mid_max:
        return admission_cfg.deadband_width["mid"]
    return admission_cfg.deadband_width["long"]


def in_band(diff: float, width: float) -> bool:
    return abs(diff) <= width
