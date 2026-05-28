`timescale 1ns / 1ps

// =============================================================================
// Module: exp_base2_converter
// Project: Flash-SFU H9-O32
// Description: Stage 1 of Custom EXP Pipeline. Converts FP16 input x to x * log2(e) ≈ x * 1.442695 using 7-term CSD Shift & Add.
// Pipeline Latency: 1 cycle
// =============================================================================

module exp_base2_converter (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  logic [15:0] x_in,
    
    output logic        valid_out,
    output logic [15:0] y_out
);

    // ====================================================================
    // 1. FP16 Decoding & Mantissa Extraction
    // ====================================================================
    logic        x_sign;
    logic [4:0]  x_exp;
    logic [9:0]  x_frac;
    
    assign x_sign = x_in[15];
    assign x_exp  = x_in[14:10];
    assign x_frac = x_in[9:0];

    // Exception handling for 0 or Subnormal (flush to 0)
    // In softmax, very small inputs or 0 inputs just yield 0 for the base2 stage,
    // though exp(0) is 1, x*log2(e) = 0.
    logic is_zero;
    assign is_zero = (x_exp == 5'b00000);

    // 11-bit mantissa including hidden '1'
    logic [10:0] m_x;
    assign m_x = is_zero ? 11'd0 : {1'b1, x_frac};
    
    // ====================================================================
    // 2. 7-term CSD Series Formulation (y = x * 1.442695)
    // ====================================================================
    // log2(e) ≈ 1 + 2^(-1) - 2^(-4) + 2^(-8) + 2^(-10) + 2^(-12) + 2^(-14)
    // We use a 16-bit CPA. Bit [14] represents 2^0 position (1.0).
    // Bit [15] is the 2^1 position (2.0 overflow check).
    // M_x is aligned such that the hidden '1' is at bit 14.
    
    // Term A: x * 2^0  (Shift 0)
    logic [15:0] term_a;
    assign term_a = {1'b0, m_x, 4'b0000};
    
    // Term B: x * 2^(-1) (Shift Right 1)
    logic [15:0] term_b;
    assign term_b = {2'b00, m_x, 3'b000};
    
    // Term C: x * 2^(-4) (Shift Right 4) -> To be subtracted (~C)
    logic [15:0] term_c_inv;
    assign term_c_inv = ~{5'b00000, m_x};
    
    // Term D: x * 2^(-8) (Shift Right 8)
    logic [15:0] term_d;
    assign term_d = {9'b000_0000_00, m_x[10:4]};
    
    // Term E: x * 2^(-10) (Shift Right 10)
    logic [15:0] term_e;
    assign term_e = {11'b000_0000_0000, m_x[10:6]};
    
    // Term F: x * 2^(-12) (Shift Right 12)
    logic [15:0] term_f;
    assign term_f = {13'b000_0000_0000_00, m_x[10:8]};

    // Term G: x * 2^(-14) (Shift Right 14)
    logic [15:0] term_g;
    assign term_g = {15'b000_0000_0000_0000, m_x[10]};

    // ====================================================================
    // 3. CSA Tree / 16-bit CPA Addition
    // ====================================================================
    logic [15:0] sum;
    
    // +16'd1 is the Two's complement correction for (~term_c_inv + 1)
    // Using behavioral '+' creates a fast CSA tree + final CPA in synthesis.
    assign sum = term_a + term_b + term_c_inv + term_d + term_e + term_f + term_g + 16'd1;

    // ====================================================================
    // 4. Zero-Cycle Round-To-Nearest (RTN)
    // ====================================================================
    logic [15:0] sum_rounded;
    logic [15:0] rtn_const;
    
    // If sum overflows to 2.0 (bit 15 == 1), fraction bits will be [14:5], so half-ULP is at bit 4.
    // If sum is < 2.0 (bit 15 == 0), fraction bits will be [13:4], so half-ULP is at bit 3.
    assign rtn_const = sum[15] ? 16'h0010 : 16'h0008; // (1<<4) or (1<<3)
    assign sum_rounded = sum + rtn_const;

    // ====================================================================
    // 5. Normalization & FP16 Exponent Re-packing
    // ====================================================================
    logic [4:0] y_exp_comb;
    logic [9:0] y_frac_comb;
    
    always_comb begin
        if (is_zero) begin
            y_exp_comb  = 5'b00000;
            y_frac_comb = 10'd0;
        end else begin
            if (sum_rounded[15]) begin
                // Overflow to 2.x
                // Shift window by 1: Leading 1 is at bit 15.
                y_exp_comb  = x_exp + 5'b00001;
                y_frac_comb = sum_rounded[14:5];
            end else begin
                // No overflow (1.x)
                // Leading 1 is at bit 14.
                y_exp_comb  = x_exp;
                y_frac_comb = sum_rounded[13:4];
            end
        end
    end

    // ====================================================================
    // 6. Output Pipeline Register (Stage 1)
    // ====================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            y_out     <= 16'd0;
        end else begin
            valid_out <= valid_in;
            // Retain the original sign according to FP16 specs.
            y_out     <= {x_sign, y_exp_comb, y_frac_comb};
        end
    end

endmodule
