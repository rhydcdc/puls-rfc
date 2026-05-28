`timescale 1ns / 1ps

// =============================================================================
// Module: fp16_to_fp32_mult
// Project: Flash-SFU H9-O32
// Description: FP16×FP16→FP32 mixed-precision multiplier (11x11=22bit, lossless)
//              No rounding needed: 22-bit product fits in FP32 23-bit mantissa
// Pipeline Latency: Combinational
// =============================================================================

module fp16_to_fp32_mult (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [31:0] result
);

    // ========================================================================
    // 1. FP16 Input Decoding
    // ========================================================================
    logic sign_a, sign_b;
    logic [4:0] exp_a, exp_b;
    logic [9:0] frac_a, frac_b;

    assign sign_a = a[15];
    assign exp_a  = a[14:10];
    assign frac_a = a[9:0];

    assign sign_b = b[15];
    assign exp_b  = b[14:10];
    assign frac_b = b[9:0];

    // Detect subnormal inputs
    logic a_is_sub, b_is_sub;
    assign a_is_sub = (exp_a == 5'h00) && (frac_a != 0);
    assign b_is_sub = (exp_b == 5'h00) && (frac_b != 0);

    // Effective exponent formulas for subnormals
    // A subnormal has an effective exponent identical to exp=1, just with a 0.xxx mantissa
    logic [4:0] eff_exp_a, eff_exp_b;
    assign eff_exp_a = (exp_a == 5'h00) ? 5'd1 : exp_a;
    assign eff_exp_b = (exp_b == 5'h00) ? 5'd1 : exp_b;

    // Mantissa extraction (implicit 1 for normal, 0 for subnormal)
    logic [10:0] ext_frac_a, ext_frac_b;
    assign ext_frac_a = {~a_is_sub & (exp_a != 0), frac_a};
    assign ext_frac_b = {~b_is_sub & (exp_b != 0), frac_b};

    // Result Sign
    logic sign_res;
    assign sign_res = sign_a ^ sign_b;

    // ========================================================================
    // 2. Exact Mantissa Multiplication (11-bit x 11-bit -> 22-bit)
    // ========================================================================
    logic [21:0] mult_res;
    assign mult_res = ext_frac_a * ext_frac_b;

    // ========================================================================
    // 3. Normalization (Leading Zero Count)
    // ========================================================================
    logic [4:0] lzc;
    
    always_comb begin
        casex (mult_res)
            22'b1xx_xxxx_xxxx_xxxx_xxxx_xxx: lzc = 5'd0;
            22'b01x_xxxx_xxxx_xxxx_xxxx_xxx: lzc = 5'd1;
            22'b001_xxxx_xxxx_xxxx_xxxx_xxx: lzc = 5'd2;
            22'b000_1xxx_xxxx_xxxx_xxxx_xxx: lzc = 5'd3;
            22'b000_01xx_xxxx_xxxx_xxxx_xxx: lzc = 5'd4;
            22'b000_001x_xxxx_xxxx_xxxx_xxx: lzc = 5'd5;
            22'b000_0001_xxxx_xxxx_xxxx_xxx: lzc = 5'd6;
            22'b000_0000_1xxx_xxxx_xxxx_xxx: lzc = 5'd7;
            22'b000_0000_01xx_xxxx_xxxx_xxx: lzc = 5'd8;
            22'b000_0000_001x_xxxx_xxxx_xxx: lzc = 5'd9;
            22'b000_0000_0001_xxxx_xxxx_xxx: lzc = 5'd10;
            22'b000_0000_0000_1xxx_xxxx_xxx: lzc = 5'd11;
            22'b000_0000_0000_01xx_xxxx_xxx: lzc = 5'd12;
            22'b000_0000_0000_001x_xxxx_xxx: lzc = 5'd13;
            22'b000_0000_0000_0001_xxxx_xxx: lzc = 5'd14;
            22'b000_0000_0000_0000_1xxx_xxx: lzc = 5'd15;
            22'b000_0000_0000_0000_01xx_xxx: lzc = 5'd16;
            22'b000_0000_0000_0000_001x_xxx: lzc = 5'd17;
            22'b000_0000_0000_0000_0001_xxx: lzc = 5'd18;
            22'b000_0000_0000_0000_0000_1xx: lzc = 5'd19;
            22'b000_0000_0000_0000_0000_01x: lzc = 5'd20;
            22'b000_0000_0000_0000_0000_001: lzc = 5'd21;
            default: lzc = 5'd22; // mult_res == 0
        endcase
    end

    // Shift mantissa so that the MSB (hidden 1) is at bit 21
    logic [21:0] shifted_mult_res;
    assign shifted_mult_res = mult_res << lzc;

    // The FP32 mantissa requires 23 fraction bits.
    // Our shifted result has the implicit '1' at bit 21, and 21 bits of fraction below it.
    // We pad with 2 zeros at the LSB to make it exactly 23 bits.
    logic [22:0] fp32_frac;
    assign fp32_frac = {shifted_mult_res[20:0], 2'b00};

    // ========================================================================
    // 4. Exact Exponent Calculation (No Overflow/Underflow Possible)
    // ========================================================================
    // Derivation:
    // Actual sum of Exponents = (eff_exp_a - 15) + (eff_exp_b - 15)
    // Shift adjustment = +1 (if lzc == 0) or 0 (if lzc == 1), so + (1 - lzc)
    // Convert to FP32 bias (+127):
    // exp_fp32 = (eff_exp_a - 15) + (eff_exp_b - 15) + 127 + 1 - lzc
    // exp_fp32 = eff_exp_a + eff_exp_b + 98 - lzc
    
    logic [7:0] fp32_exp;
    assign fp32_exp = {3'b000, eff_exp_a} + {3'b000, eff_exp_b} + 8'd98 - {3'b000, lzc};

    // ========================================================================
    // 5. Special Cases & Final Output
    // ========================================================================
    logic a_isNaN, b_isNaN, a_isInf, b_isInf, a_isZero, b_isZero;
    assign a_isNaN  = (exp_a == 5'h1f) && (frac_a != 10'd0);
    assign b_isNaN  = (exp_b == 5'h1f) && (frac_b != 10'd0);
    assign a_isInf  = (exp_a == 5'h1f) && (frac_a == 10'd0);
    assign b_isInf  = (exp_b == 5'h1f) && (frac_b == 10'd0);
    assign a_isZero = (exp_a == 5'h00) && (frac_a == 10'd0);
    assign b_isZero = (exp_b == 5'h00) && (frac_b == 10'd0);

    always_comb begin
        if (a_isNaN || b_isNaN) begin
            result = 32'h7FC00000; // Quiet NaN
        end else if ((a_isInf && b_isZero) || (a_isZero && b_isInf)) begin
            result = 32'h7FC00000; // NaN (Inf * 0)
        end else if (a_isInf || b_isInf) begin
            result = {sign_res, 8'hFF, 23'h000000}; // Inf
        end else if (a_isZero || b_isZero) begin
            result = {sign_res, 31'h00000000}; // Zero
        end else begin
            result = {sign_res, fp32_exp, fp32_frac}; // Normal Result
        end
    end

endmodule
