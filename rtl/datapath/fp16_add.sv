`timescale 1ns / 1ps

// =============================================================================
// Module: fp16_add
// Project: Flash-SFU H9-O32
// Description: FP16 Adder with RTNE rounding and subnormal handling
// Pipeline Latency: Combinational
// =============================================================================

module fp16_add (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] result
);

    logic sign_a, sign_b;
    logic [4:0] exp_a, exp_b;
    logic [9:0] frac_a, frac_b;
    
    assign sign_a = a[15];
    assign exp_a  = a[14:10];
    assign frac_a = a[9:0];
    
    assign sign_b = b[15];
    assign exp_b  = b[14:10];
    assign frac_b = b[9:0];
    
    // Subnormal handling
    logic [4:0] eff_exp_a, eff_exp_b;
    logic a_is_sub, b_is_sub;
    assign a_is_sub = (exp_a == 5'h00);
    assign b_is_sub = (exp_b == 5'h00);
    
    assign eff_exp_a = a_is_sub ? 5'd1 : exp_a;
    assign eff_exp_b = b_is_sub ? 5'd1 : exp_b;
    
    logic [10:0] ext_frac_a, ext_frac_b;
    assign ext_frac_a = {~a_is_sub, frac_a};
    assign ext_frac_b = {~b_is_sub, frac_b};
    
    // Compare magnitudes
    logic a_larger;
    assign a_larger = (eff_exp_a > eff_exp_b) || ((eff_exp_a == eff_exp_b) && (ext_frac_a >= ext_frac_b));
    
    logic sign_large, sign_small;
    logic [4:0] exp_large, exp_small;
    logic [10:0] frac_large, frac_small;
    
    assign sign_large = a_larger ? sign_a : sign_b;
    assign sign_small = a_larger ? sign_b : sign_a;
    assign exp_large  = a_larger ? eff_exp_a : eff_exp_b;
    assign exp_small  = a_larger ? eff_exp_b : eff_exp_a;
    assign frac_large = a_larger ? ext_frac_a : ext_frac_b;
    assign frac_small = a_larger ? ext_frac_b : ext_frac_a;
    
    // Exponent difference
    logic [4:0] exp_diff;
    assign exp_diff = exp_large - exp_small;
    
    // Alignment shift (keep 3 extra bits: G, R, S)
    logic [23:0] shifted_frac_small;
    assign shifted_frac_small = {frac_small, 13'b0} >> (exp_diff > 14 ? 14 : exp_diff);
    
    logic [13:0] aligned_small; // 11 bit + G + R + S
    assign aligned_small = {shifted_frac_small[23:13], shifted_frac_small[12], shifted_frac_small[11], |shifted_frac_small[10:0]};
    
    logic [13:0] aligned_large;
    assign aligned_large = {frac_large, 3'b000};
    
    // Add/Sub
    logic sub_op;
    assign sub_op = (sign_large != sign_small);
    
    logic [14:0] sum_raw;
    assign sum_raw = sub_op ? (aligned_large - aligned_small) : (aligned_large + aligned_small);
    
    // Normalization
    logic [4:0] lzc;
    always_comb begin
        if (sum_raw[14]) lzc = 5'd0;
        else if (sum_raw[13]) lzc = 5'd1;
        else if (sum_raw[12]) lzc = 5'd2;
        else if (sum_raw[11]) lzc = 5'd3;
        else if (sum_raw[10]) lzc = 5'd4;
        else if (sum_raw[9])  lzc = 5'd5;
        else if (sum_raw[8])  lzc = 5'd6;
        else if (sum_raw[7])  lzc = 5'd7;
        else if (sum_raw[6])  lzc = 5'd8;
        else if (sum_raw[5])  lzc = 5'd9;
        else if (sum_raw[4])  lzc = 5'd10;
        else if (sum_raw[3])  lzc = 5'd11;
        else if (sum_raw[2])  lzc = 5'd12;
        else if (sum_raw[1])  lzc = 5'd13;
        else if (sum_raw[0])  lzc = 5'd14;
        else lzc = 5'd15;
    end
    
    logic [4:0] exp_adjusted;
    logic [14:0] sum_shifted;
    
    always_comb begin
        if (exp_large >= lzc) begin
            exp_adjusted = exp_large - lzc + 1;
            sum_shifted = sum_raw << lzc;
        end else begin
            exp_adjusted = 5'd0; // Subnormal
            sum_shifted = sum_raw << exp_large;
        end
    end
    
    // Round to Nearest Even
    logic round_up;
    // sum_shifted[13:4] is the 10-bit fraction.
    // sum_shifted[3] is Guard, [2] is Round, [1:0] is Sticky.
    assign round_up = sum_shifted[3] & (sum_shifted[2] | sum_shifted[1] | sum_shifted[0] | sum_shifted[4]); 
    
    logic [10:0] final_frac_raw;
    // Add round_up to the 10-bit fraction. 11th bit captures fraction overflow (carry).
    assign final_frac_raw = sum_shifted[13:4] + round_up;
    
    logic [4:0] final_exp;
    logic [9:0] final_frac;
    
    always_comb begin
        final_exp = exp_adjusted;
        final_frac = final_frac_raw[9:0];
        
        if (final_frac_raw[10]) begin // Fraction overflowed from 3FF to 400
            if (exp_adjusted == 5'd0) begin 
                final_exp = 5'd1; // Becomes normalized min value
                final_frac = 10'd0;
            end else begin
                final_exp = exp_adjusted + 1;
                final_frac = 10'd0;
            end
        end
    end
    
    // Special Cases
    logic a_isNaN, b_isNaN, a_isInf, b_isInf, a_isZero, b_isZero;
    assign a_isNaN = (exp_a == 5'h1f) && (frac_a != 10'd0);
    assign b_isNaN = (exp_b == 5'h1f) && (frac_b != 10'd0);
    assign a_isInf = (exp_a == 5'h1f) && (frac_a == 10'd0);
    assign b_isInf = (exp_b == 5'h1f) && (frac_b == 10'd0);
    assign a_isZero = (exp_a == 5'h00) && (frac_a == 10'd0);
    assign b_isZero = (exp_b == 5'h00) && (frac_b == 10'd0);
    
    always_comb begin
        if (a_isNaN || b_isNaN) begin
            result = 16'h7E00; // NaN
        end else if (a_isInf && b_isInf) begin
            if (sign_a == sign_b) result = {sign_a, 15'h7C00}; // Inf
            else result = 16'h7E00; // NaN
        end else if (a_isInf) begin
            result = {sign_a, 15'h7C00};
        end else if (b_isInf) begin
            result = {sign_b, 15'h7C00};
        end else if (sum_raw == 15'd0) begin
            result = { (sign_a & sign_b), 15'd0 }; // -0 if both are -0
        end else if (final_exp >= 5'h1f) begin
            result = {sign_large, 15'h7C00}; // Overflow to Inf
        end else begin
            result = {sign_large, final_exp, final_frac};
        end
    end

endmodule
