`timescale 1ns / 1ps

// =============================================================================
// Module: fp32_add
// Project: Flash-SFU H9-O32
// Description: FP32 Adder with RTNE rounding and subnormal handling
// Pipeline Latency: Combinational
// =============================================================================

module fp32_add (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] result
);

    logic sign_a, sign_b;
    logic [7:0] exp_a, exp_b;
    logic [22:0] frac_a, frac_b;
    
    assign sign_a = a[31];
    assign exp_a  = a[30:23];
    assign frac_a = a[22:0];
    
    assign sign_b = b[31];
    assign exp_b  = b[30:23];
    assign frac_b = b[22:0];
    
    // Subnormal handling
    logic [7:0] eff_exp_a, eff_exp_b;
    logic a_is_sub, b_is_sub;
    assign a_is_sub = (exp_a == 8'h00);
    assign b_is_sub = (exp_b == 8'h00);
    
    assign eff_exp_a = a_is_sub ? 8'd1 : exp_a;
    assign eff_exp_b = b_is_sub ? 8'd1 : exp_b;
    
    logic [23:0] ext_frac_a, ext_frac_b;
    assign ext_frac_a = {~a_is_sub, frac_a};
    assign ext_frac_b = {~b_is_sub, frac_b};
    
    // Compare magnitudes
    logic a_larger;
    assign a_larger = (eff_exp_a > eff_exp_b) || ((eff_exp_a == eff_exp_b) && (ext_frac_a >= ext_frac_b));
    
    logic sign_large, sign_small;
    logic [7:0] exp_large, exp_small;
    logic [23:0] ext_frac_large, ext_frac_small;
    
    assign sign_large     = a_larger ? sign_a : sign_b;
    assign sign_small     = a_larger ? sign_b : sign_a;
    assign exp_large      = a_larger ? eff_exp_a : eff_exp_b;
    assign exp_small      = a_larger ? eff_exp_b : eff_exp_a;
    assign ext_frac_large = a_larger ? ext_frac_a : ext_frac_b;
    assign ext_frac_small = a_larger ? ext_frac_b : ext_frac_a;
    
    // Exponent difference
    logic [7:0] exp_diff;
    assign exp_diff = exp_large - exp_small;
    
    // Alignment shift (keep 3 extra bits: G, R, S)
    logic [49:0] shifted_frac_small;
    assign shifted_frac_small = {ext_frac_small, 26'b0} >> (exp_diff > 8'd26 ? 8'd26 : exp_diff);
    
    // 24 bit + G + R + S = 27 bit
    logic [26:0] aligned_small; 
    assign aligned_small = {shifted_frac_small[49:26], shifted_frac_small[25], shifted_frac_small[24], |shifted_frac_small[23:0]};
    
    logic [26:0] aligned_large;
    assign aligned_large = {ext_frac_large, 3'b000};
    
    // Add/Sub
    logic sub_op;
    assign sub_op = (sign_large != sign_small);
    
    logic [27:0] sum_raw;
    assign sum_raw = sub_op ? (aligned_large - aligned_small) : (aligned_large + aligned_small);
    
    // Normalization logic (Leading Zero Counter 28-bit deep)
    logic [4:0] lzc;
    always_comb begin
        if (sum_raw[27]) lzc = 5'd0;
        else if (sum_raw[26]) lzc = 5'd1;
        else if (sum_raw[25]) lzc = 5'd2;
        else if (sum_raw[24]) lzc = 5'd3;
        else if (sum_raw[23]) lzc = 5'd4;
        else if (sum_raw[22]) lzc = 5'd5;
        else if (sum_raw[21]) lzc = 5'd6;
        else if (sum_raw[20]) lzc = 5'd7;
        else if (sum_raw[19]) lzc = 5'd8;
        else if (sum_raw[18]) lzc = 5'd9;
        else if (sum_raw[17]) lzc = 5'd10;
        else if (sum_raw[16]) lzc = 5'd11;
        else if (sum_raw[15]) lzc = 5'd12;
        else if (sum_raw[14]) lzc = 5'd13;
        else if (sum_raw[13]) lzc = 5'd14;
        else if (sum_raw[12]) lzc = 5'd15;
        else if (sum_raw[11]) lzc = 5'd16;
        else if (sum_raw[10]) lzc = 5'd17;
        else if (sum_raw[9])  lzc = 5'd18;
        else if (sum_raw[8])  lzc = 5'd19;
        else if (sum_raw[7])  lzc = 5'd20;
        else if (sum_raw[6])  lzc = 5'd21;
        else if (sum_raw[5])  lzc = 5'd22;
        else if (sum_raw[4])  lzc = 5'd23;
        else if (sum_raw[3])  lzc = 5'd24;
        else if (sum_raw[2])  lzc = 5'd25;
        else if (sum_raw[1])  lzc = 5'd26;
        else if (sum_raw[0])  lzc = 5'd27;
        else lzc = 5'd28;
    end
    
    logic [7:0] exp_adjusted;
    logic [27:0] sum_shifted;
    
    always_comb begin
        if (exp_large >= lzc) begin
            exp_adjusted = exp_large - lzc + 1;
            sum_shifted = sum_raw << lzc;
        end else begin
            exp_adjusted = 8'd0; // Subnormal Fallback
            sum_shifted = sum_raw << exp_large;
        end
    end
    
    // Round to Nearest Even (RTN)
    logic round_up;
    // sum_shifted[26:4] is the 23-bit fraction.
    // sum_shifted[3] is Guard, [2] is Round, [1:0] is Sticky.
    assign round_up = sum_shifted[3] & (sum_shifted[2] | sum_shifted[1] | sum_shifted[0] | sum_shifted[4]); 
    
    logic [23:0] final_frac_raw;
    // Add round_up to the 23-bit fraction. 24th bit captures fraction overflow (carry).
    assign final_frac_raw = sum_shifted[26:4] + round_up;
    
    logic [7:0] final_exp;
    logic [22:0] final_frac;
    
    always_comb begin
        final_exp = exp_adjusted;
        final_frac = final_frac_raw[22:0];
        
        if (final_frac_raw[23]) begin // Fraction overflowed from 7FFFFF to 800000
            if (exp_adjusted == 8'd0) begin 
                final_exp = 8'd1; // Becomes normalized min value
                final_frac = 23'd0;
            end else begin
                final_exp = exp_adjusted + 1;
                final_frac = 23'd0;
            end
        end
    end
    
    // Special Cases
    logic a_isNaN, b_isNaN, a_isInf, b_isInf, a_isZero, b_isZero;
    assign a_isNaN = (exp_a == 8'hFF) && (frac_a != 23'd0);
    assign b_isNaN = (exp_b == 8'hFF) && (frac_b != 23'd0);
    assign a_isInf = (exp_a == 8'hFF) && (frac_a == 23'd0);
    assign b_isInf = (exp_b == 8'hFF) && (frac_b == 23'd0);
    assign a_isZero = (exp_a == 8'h00) && (frac_a == 23'd0);
    assign b_isZero = (exp_b == 8'h00) && (frac_b == 23'd0);
    
    always_comb begin
        if (a_isNaN || b_isNaN) begin
            result = 32'h7FC00000; // NaN
        end else if (a_isInf && b_isInf) begin
            if (sign_a == sign_b) result = {sign_a, 31'h7F800000}; // Inf
            else result = 32'h7FC00000; // NaN (Inf - Inf)
        end else if (a_isInf) begin
            result = {sign_a, 31'h7F800000};
        end else if (b_isInf) begin
            result = {sign_b, 31'h7F800000};
        end else if (sum_raw == 28'd0) begin
            result = { (sign_a & sign_b), 31'd0 }; // -0 if both are -0
        end else if (final_exp >= 8'hFF) begin
            result = {sign_large, 31'h7F800000}; // Overflow to Inf
        end else begin
            result = {sign_large, final_exp, final_frac};
        end
    end

endmodule
