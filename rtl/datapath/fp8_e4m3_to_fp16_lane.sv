`timescale 1ns / 1ps

// =============================================================================
// Module: fp8_e4m3_to_fp16_lane
// Project: Flash-SFU H9-O32 (FP8 KV Cache extension)
// Description: Single-lane FP8 E4M3 → FP16 dequant.
//              OCP OFP8 v1.0 §5.1 정합:
//                - Normal:   v = (-1)^S × 2^(E-7) × (1 + M/8)
//                - Subnormal: v = (-1)^S × 2^-6 × (M/8)   [E=0, M∈{1..7}]
//                - Zero:     S.0000.000
//                - NaN:      S.1111.111  (single bit pattern)
//              FP16 mapping (saturating; subnormal → FP16 normal via 7-entry LUT):
//                - Normal     → exp_fp16 = exp_e4m3 + 8 (rebias 7→15), mant zero-pad
//                - Subnormal  → 7-entry LUT (per design doc §2.6.4)
//                - NaN/Zero   → IEEE 754 FP16 NaN/Zero
// Pipeline Latency: Combinational (0 cycle)
// =============================================================================

module fp8_e4m3_to_fp16_lane (
    input  logic [7:0]  fp8_in,
    output logic [15:0] fp16_out
);

    // -------------------------------------------------------------------------
    // 1. Bit field decode
    // -------------------------------------------------------------------------
    logic       sign;
    logic [3:0] exp_e4m3;
    logic [2:0] mant_e4m3;

    assign sign      = fp8_in[7];
    assign exp_e4m3  = fp8_in[6:3];
    assign mant_e4m3 = fp8_in[2:0];

    // -------------------------------------------------------------------------
    // 2. Special / range classification
    // -------------------------------------------------------------------------
    logic is_zero;
    logic is_nan;
    logic is_subn;
    logic is_normal;

    assign is_zero   = (exp_e4m3 == 4'd0)  && (mant_e4m3 == 3'd0);
    assign is_nan    = (exp_e4m3 == 4'd15) && (mant_e4m3 == 3'd7);
    assign is_subn   = (exp_e4m3 == 4'd0)  && (mant_e4m3 != 3'd0);
    assign is_normal = ~(is_zero | is_nan | is_subn);

    // -------------------------------------------------------------------------
    // 3. Normal-path mapping
    //    FP16 exponent = exp_e4m3 + 8  (bias 7 → 15)
    //    FP16 mantissa = {mant_e4m3, 7'b0}  (3-bit pad to 10-bit)
    // -------------------------------------------------------------------------
    logic [4:0]  exp_fp16_n;
    logic [9:0]  mant_fp16_n;
    logic [15:0] fp16_normal;

    assign exp_fp16_n  = {1'b0, exp_e4m3} + 5'd8;
    assign mant_fp16_n = {mant_e4m3, 7'b0};
    assign fp16_normal = {sign, exp_fp16_n, mant_fp16_n};

    // -------------------------------------------------------------------------
    // 4. Subnormal-path 7-entry LUT
    //    m=1: 2^-9              → FP16 (e=6, m=0)
    //    m=2: 2^-8              → FP16 (e=7, m=0)
    //    m=3: 1.5 × 2^-8        → FP16 (e=7, m=512)
    //    m=4: 2^-7              → FP16 (e=8, m=0)
    //    m=5: 1.25 × 2^-7       → FP16 (e=8, m=256)
    //    m=6: 1.5 × 2^-7        → FP16 (e=8, m=512)
    //    m=7: 1.75 × 2^-7       → FP16 (e=8, m=768)
    // -------------------------------------------------------------------------
    logic [14:0] subn_mag;
    logic [15:0] fp16_subn;

    always_comb begin
        unique case (mant_e4m3)
            3'd1:    subn_mag = {5'd6, 10'd0};            // 0x1800
            3'd2:    subn_mag = {5'd7, 10'd0};            // 0x1C00
            3'd3:    subn_mag = {5'd7, 10'd512};          // 0x1E00
            3'd4:    subn_mag = {5'd8, 10'd0};            // 0x2000
            3'd5:    subn_mag = {5'd8, 10'd256};          // 0x2100
            3'd6:    subn_mag = {5'd8, 10'd512};          // 0x2200
            3'd7:    subn_mag = {5'd8, 10'd768};          // 0x2300
            default: subn_mag = 15'd0;                    // m=0 unreachable here
        endcase
    end

    assign fp16_subn = {sign, subn_mag};

    // -------------------------------------------------------------------------
    // 5. Special-value constants
    //    FP16 Zero : sign-only, exp=0, mant=0
    //    FP16 NaN  : exp=11111, mant ≠ 0  (canonical quiet NaN: mant[9]=1)
    // -------------------------------------------------------------------------
    logic [15:0] fp16_zero;
    logic [15:0] fp16_nan;

    assign fp16_zero = {sign, 5'd0, 10'd0};
    assign fp16_nan  = {sign, 5'b11111, 10'b1000000000};

    // -------------------------------------------------------------------------
    // 6. Output 4-way mux (priority: zero > nan > subnormal > normal)
    // -------------------------------------------------------------------------
    always_comb begin
        if      (is_zero)  fp16_out = fp16_zero;
        else if (is_nan)   fp16_out = fp16_nan;
        else if (is_subn)  fp16_out = fp16_subn;
        else               fp16_out = fp16_normal;
    end

endmodule
