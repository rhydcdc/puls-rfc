"""KV 트랜잭션 감소 + 믹스드 배치 FFN 가중치 절감 → 버스 트래픽(바이트) 감소량 +
시스템 발열(에너지) 절감 배수 산출.

사용자 결정 (대화 정합):
- 트랜잭션 단위 = 바이트 (전송폭 8 B = 표 Memory 열 "(64bit)" 헤더 → access 환산)
- DRAM 접근 에너지 = 밴드 처리 (1.3 / 1.95 / 2.6 nJ per access)
- 연산 floor = FP16 MAC = FMult16(1.1) + FAdd16(0.4) = 1.5 pJ  (표 밖 가정 0)
- 배수 = 절감 배수만 (E_baseline / E_PULS)
- 두 시나리오 분리 산출:
    (A) 순수 decode step — KV 트랜잭션 제거 효과 단독 (연산 floor 작음 → KV 지배)
    (B) 믹스드 배치 포함 — KV 제거 + FFN 가중치 절반 합산 (연산강도 상승, FFN 연산이 floor 키움)

상수 출처:
- 에너지 상수 = Horowitz Fig 1.1.9 (45nm 0.9V) — FP16 FAdd 0.4 pJ, FMult 1.1 pJ,
  DRAM access 1.3-2.6 nJ, Memory 열 "(64bit)" = 8 B/access. (사용자 첨부 표)
- 모델/기판 상수 = puls_sched.config (Llama-3 70B + HBM4 projection, 리포 calibrated)

발열 모델 (시스템 배수):
    E_total = E_bus(외부 GPU↔HBM 트랜잭션) + E_compute(MAC floor, 양측 동일)
    E_bus     = bytes × e_dram_per_byte
    E_compute = MAC수 × 1.5 pJ                  ← floor (분모를 0 으로부터 받침, base=PuLS 동일)
    발열 절감 배수 = E_total_baseline / E_total_PULS

전제 (RFC 주장 그대로 — 독립 검증 아님): PIM 인-다이 attention 은 외부 버스
트랜잭션을 발생시키지 않음 → PuLS 의 KV 외부 버스 = 0. 본 스크립트는 이 전제 위에서
수치를 산출한다 (전제 자체를 증명하지 않음).
"""

import csv
import sys
from pathlib import Path

# Windows 콘솔 cp949 → UTF-8 강제 (유니코드 기호 출력)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 리포 calibrated 상수 재사용 (DRY — config.py 단일 출처)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from puls_sched.config import default_dummy_config  # noqa: E402

cfg = default_dummy_config()
M = cfg.model
CAL = cfg.calibration

# ============================================================================
# 1. 상수
# ============================================================================

# --- Horowitz Fig 1.1.9 (사용자 첨부 표) ---
E_FADD16_PJ = 0.4
E_FMULT16_PJ = 1.1
E_MAC_FP16_PJ = E_FMULT16_PJ + E_FADD16_PJ          # = 1.5 pJ / MAC (연산 floor)
DRAM_ACCESS_NJ_BAND = {"low": 1.3, "mid": 1.95, "high": 2.6}   # 1.95 = (1.3+2.6)/2 헤드라인
BURST_BYTES = 8                                      # 표 Memory 열 헤더 "(64bit)" = 8 B / access

def e_dram_per_byte_pj(nj_per_access: float) -> float:
    """DRAM 접근 에너지(nJ/access) → per-byte(pJ/B). access = 64bit = 8 B (표 헤더)."""
    return nj_per_access * 1000.0 / BURST_BYTES      # nJ→pJ(×1000), ÷8 B

E_DRAM_PB = {k: e_dram_per_byte_pj(v) for k, v in DRAM_ACCESS_NJ_BAND.items()}

# --- 모델 / 기판 (config.py) ---
L = M.num_layers                                     # 80
H = M.hidden                                         # 8192
N_Q = M.num_heads                                    # 64
N_KV = M.num_kv_heads                                # 8
D = M.head_dim                                       # 128
FF = M.ffn_intermediate                              # 28672
KV_BYTES_TOK_LAYER = CAL.kv_bytes_per_token_per_layer_fp8   # 2048 (FP8: 2×8×128×1B)
FFN_BYTES_LAYER = 3 * H * FF * 2                     # SwiGLU 3 GEMM × FP16 = 1.409e9

# --- per-token / per-layer MAC 수 (연산 floor, 양측 동일) ---
MAC_PROJ_TOK_LAYER = H * (H + 2 * N_KV * D) + H * H  # QKV + O projection
MAC_FFN_TOK_LAYER = 3 * H * FF                       # SwiGLU 3 GEMM
def mac_decode_attn_tok_layer(ctx: int) -> int:
    """디코드 1 토큰 attention MAC/layer = QK^T + AV = 2 × n_q × d × ctx."""
    return 2 * N_Q * D * ctx

def acc(b: float) -> float:
    return b / BURST_BYTES

# ============================================================================
# 2. 워크로드 — 트레이스 실제 KV 길이 (LongBench λ=3.40)
# ============================================================================

TRACE = Path(__file__).parent.parent / "data" / "longctx_longbench_lambda_3_40.csv"
N_DECODE_REQ = 3            # README runtime validation 정합 (첫 3 req: 47K·280K·81K)
PREFILL_CHUNK = 512         # 믹스드 배치에 함께 태우는 prefill chunk (admission default)

def load_ctx_lengths(n: int) -> list[int]:
    with open(TRACE, newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(rows[i]["num_prefill_tokens"]) for i in range(n)]

CTX = load_ctx_lengths(N_DECODE_REQ)        # 디코드 req 들의 KV 길이
B_D = len(CTX)                              # 디코드 배치 크기

# ============================================================================
# 3. 시나리오 엔진 — 버스 바이트 + 연산 floor → 발열 배수
# ============================================================================

def scenario(name: str, *, prefill_chunk: int, ffn_weight_reuse: bool) -> dict:
    """한 시나리오의 버스 바이트(KV·FFN가중치) + 연산 floor(MAC) 산출.

    Args:
        prefill_chunk: 믹스드 배치에 태우는 prefill 토큰 수 (0 = 순수 decode)
        ffn_weight_reuse: True면 PuLS가 FFN 가중치 1× 로드 (믹스드), False면 baseline과 동일 2×
            (순수 decode 에서는 prefill 배치가 없으므로 가중치 재사용 효과 자체가 없음 → False)
    """
    t_tok = B_D + prefill_chunk            # FFN/proj 를 통과하는 총 토큰

    # --- 버스 바이트 (디코드 1 step, 전 layer) ---
    # KV 버스: baseline 은 외부에서 전 KV 읽음, PuLS 는 PIM 인-다이 → result 잔여만
    kv_base = sum(ctx * KV_BYTES_TOK_LAYER * L for ctx in CTX)
    result_puls = B_D * H * 2 * L           # PIM→HBM→GPU result [B×hidden] FP16
    kv_puls = result_puls

    # FFN 가중치 버스
    if ffn_weight_reuse:
        ffn_base = 2 * FFN_BYTES_LAYER * L  # prefill+decode 따로 → 2× 로드
        ffn_puls = 1 * FFN_BYTES_LAYER * L  # 믹스드 1× 로드 (Aux1)
    else:
        ffn_base = 1 * FFN_BYTES_LAYER * L  # 순수 decode: 가중치 1× (재사용 효과 없음)
        ffn_puls = 1 * FFN_BYTES_LAYER * L

    bytes_base = kv_base + ffn_base
    bytes_puls = kv_puls + ffn_puls

    # --- 연산 floor (MAC × 1.5 pJ), baseline = PuLS 동일 ---
    mac_attn = sum(mac_decode_attn_tok_layer(ctx) for ctx in CTX) * L
    mac_proj = t_tok * MAC_PROJ_TOK_LAYER * L
    mac_ffn = t_tok * MAC_FFN_TOK_LAYER * L
    mac_total = mac_attn + mac_proj + mac_ffn
    e_compute = mac_total * E_MAC_FP16_PJ

    # --- DRAM 밴드별 발열 배수 ---
    bands = {}
    for k, nj in DRAM_ACCESS_NJ_BAND.items():
        e_pb = E_DRAM_PB[k]
        e_bus_base = bytes_base * e_pb
        e_bus_puls = bytes_puls * e_pb
        e_base = e_bus_base + e_compute
        e_puls = e_bus_puls + e_compute
        bands[k] = {
            "nj": nj, "e_pb": e_pb,
            "e_bus_base_uJ": e_bus_base / 1e6, "e_bus_puls_uJ": e_bus_puls / 1e6,
            "e_base_uJ": e_base / 1e6, "e_puls_uJ": e_puls / 1e6,
            "mult": e_base / e_puls, "red_pct": (1 - e_puls / e_base) * 100,
        }
    return {
        "name": name, "prefill_chunk": prefill_chunk, "t_tok": t_tok,
        "kv_base": kv_base, "kv_puls": kv_puls,
        "ffn_base": ffn_base, "ffn_puls": ffn_puls,
        "bytes_base": bytes_base, "bytes_puls": bytes_puls,
        "mac_attn": mac_attn, "mac_proj": mac_proj, "mac_ffn": mac_ffn,
        "mac_total": mac_total, "e_compute_uJ": e_compute / 1e6,
        "bands": bands,
    }

# ============================================================================
# PART 1 — 트랜잭션 (바이트) 감소  [밴드 무관: 순수 바이트비]
# ============================================================================

# 트랜잭션 절감은 믹스드 시나리오 기준 (KV + FFN 가중치 둘 다 보임)
B = scenario("(B) 믹스드 배치 포함", prefill_chunk=PREFILL_CHUNK, ffn_weight_reuse=True)

print("=" * 80)
print(" PART 1 — 트랜잭션 (바이트 / access) 감소  [밴드 무관: 순수 바이트비]")
print("=" * 80)
print(f" 워크로드: 디코드 {B_D} req KV={CTX} + prefill chunk {PREFILL_CHUNK}, L={L}")
print()
print(" [KV 버스] 디코드 1 step, 외부 GPU↔HBM")
print(f"   baseline : {B['kv_base']:,.0f} B  = {acc(B['kv_base']):,.0f} access (8B/64bit)")
print(f"   PuLS     : {B['kv_puls']:,.0f} B  = {acc(B['kv_puls']):,.0f} access  (KV 외부버스 0, result 잔여)")
print(f"   감소 배수: {B['kv_base'] / B['kv_puls']:,.1f}×  "
      f"({(1 - B['kv_puls'] / B['kv_base']) * 100:.2f}% 감소)")
print()
print(" [FFN 가중치 버스] 디코드 1 step (믹스드: prefill+decode 가중치 공유)")
print(f"   baseline : {B['ffn_base']:,.0f} B  (2× 로드)")
print(f"   PuLS     : {B['ffn_puls']:,.0f} B  (1× 로드)")
print(f"   감소 배수: {B['ffn_base'] / B['ffn_puls']:.2f}×  (50.00% 감소)")
print(f"   (바이트는 정확히 2.0×. 시간 배수는 roofline 위 T4 측정 1.97× — config 정합)")
print()
print(" [KV + FFN가중치 합산 버스]")
print(f"   감소 배수: {B['bytes_base'] / B['bytes_puls']:,.1f}×  "
      f"({(1 - B['bytes_puls'] / B['bytes_base']) * 100:.2f}% 감소)")

# ============================================================================
# PART 2 — 발열 (에너지) 절감 배수  [시스템 배수 = 버스 + 연산 floor]
# ============================================================================

A = scenario("(A) 순수 decode step", prefill_chunk=0, ffn_weight_reuse=False)

def print_heat(s: dict) -> None:
    print()
    print("-" * 80)
    print(f" {s['name']}  (prefill_chunk={s['prefill_chunk']}, 총 {s['t_tok']} tok/step)")
    print("-" * 80)
    print(f"   연산 floor: MAC {s['mac_total']:,.0f} × 1.5 pJ = {s['e_compute_uJ']:,.2f} µJ  (base=PuLS 동일)")
    print(f"     ├ decode-attn: {s['mac_attn']:,.0f}   ├ proj: {s['mac_proj']:,.0f}   └ FFN: {s['mac_ffn']:,.0f}")
    print(f"   {'DRAM(nJ)':<9} {'e_dram(pJ/B)':<13} {'E_base(µJ)':<13} {'E_PuLS(µJ)':<13} {'절감배수':<9} 절감%")
    for k in ("low", "mid", "high"):
        bd = s["bands"][k]
        tag = "  ← 헤드라인" if k == "mid" else ""
        print(f"   {bd['nj']:<9} {bd['e_pb']:<13.2f} {bd['e_base_uJ']:<13,.1f} "
              f"{bd['e_puls_uJ']:<13,.1f} {bd['mult']:<9.2f} {bd['red_pct']:.1f}%{tag}")

print()
print("=" * 80)
print(" PART 2 — 발열 (에너지) 절감 배수  [시스템 배수 = 버스 + 연산 floor]")
print("=" * 80)
print_heat(A)
print_heat(B)

# ============================================================================
# 요약
# ============================================================================
print()
print("=" * 80)
print(" 요약")
print("=" * 80)
print(f"  • KV 트랜잭션 감소        : {B['kv_base'] / B['kv_puls']:,.0f}× "
      f"({(1 - B['kv_puls'] / B['kv_base']) * 100:.1f}% 감소)")
print(f"  • FFN 가중치 트랜잭션 감소 : 2.0× (50%, 바이트 기준)")
print(f"  • 버스 합산 트래픽 감소    : {B['bytes_base'] / B['bytes_puls']:,.1f}×")
print()
print(f"  • 발열 절감 배수 (A) 순수 decode  : {A['bands']['mid']['mult']:.2f}×  "
      f"밴드 [{A['bands']['low']['mult']:.2f}×–{A['bands']['high']['mult']:.2f}×]  "
      f"← KV 트랜잭션 제거 효과 단독")
print(f"  • 발열 절감 배수 (B) 믹스드 포함   : {B['bands']['mid']['mult']:.2f}×  "
      f"밴드 [{B['bands']['low']['mult']:.2f}×–{B['bands']['high']['mult']:.2f}×]  "
      f"← KV 제거 + FFN 가중치 절반 (FFN 연산이 floor 키워 희석)")
print("  (헤드라인 = DRAM 1.95 nJ; 밴드 방향: DRAM 에너지 ↑ → 버스 지배 ↑ → 절감 배수 ↑)")
