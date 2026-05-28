`timescale 1ns / 1ps

// =============================================================================
// Module: fp32_mult
// Project: Flash-SFU H9-O32
// Description: FP32 Multiplier (24x24=48bit) with RTNE rounding
//              Core arithmetic unit for Horner MAC, X*O_prev, X*l_prev
// Pipeline Latency: Combinational
// =============================================================================

module fp32_mult (
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

    logic a_is_sub, b_is_sub;
    assign a_is_sub = (exp_a == 8'h00);
    assign b_is_sub = (exp_b == 8'h00);

    logic [7:0] eff_exp_a, eff_exp_b;
    assign eff_exp_a = a_is_sub ? 8'd1 : exp_a;
    assign eff_exp_b = b_is_sub ? 8'd1 : exp_b;

    logic [23:0] ext_frac_a, ext_frac_b;
    assign ext_frac_a = {~a_is_sub, frac_a};
    assign ext_frac_b = {~b_is_sub, frac_b};

    logic sign_res;
    assign sign_res = sign_a ^ sign_b;

    // Multiplication
    logic [47:0] mult_res;
    assign mult_res = ext_frac_a * ext_frac_b; 

    // Exponent addition
    logic signed [9:0] exp_sum;
    assign exp_sum = $signed({2'b0, eff_exp_a}) + $signed({2'b0, eff_exp_b}) - 10'sd126;

    // Normalization
    logic [4:0] lzc;
    
    always_comb begin
        if (mult_res[47]) lzc = 5'd0;
        else if (mult_res[46]) lzc = 5'd1;
        else if (mult_res[45]) lzc = 5'd2;
        else if (mult_res[44]) lzc = 5'd3;
        else if (mult_res[43]) lzc = 5'd4;
        else if (mult_res[42]) lzc = 5'd5;
        else if (mult_res[41]) lzc = 5'd6;
        else if (mult_res[40]) lzc = 5'd7;
        else if (mult_res[39]) lzc = 5'd8;
        else if (mult_res[38]) lzc = 5'd9;
        else if (mult_res[37]) lzc = 5'd10;
        else if (mult_res[36]) lzc = 5'd11;
        else if (mult_res[35]) lzc = 5'd12;
        else if (mult_res[34]) lzc = 5'd13;
        else if (mult_res[33]) lzc = 5'd14;
        else if (mult_res[32]) lzc = 5'd15;
        else if (mult_res[31]) lzc = 5'd16;
        else if (mult_res[30]) lzc = 5'd17;
        else if (mult_res[29]) lzc = 5'd18;
        else if (mult_res[28]) lzc = 5'd19;
        else if (mult_res[27]) lzc = 5'd20;
        else if (mult_res[26]) lzc = 5'd21;
        else if (mult_res[25]) lzc = 5'd22;
        else if (mult_res[24]) lzc = 5'd23;
        else lzc = 5'd24; 
    end
    
    logic [55:0] shifted_frac; 
    logic signed [9:0] norm_exp;
    logic [7:0] final_exp;
    logic [22:0] final_mantissa;
    logic guard_bit, round_bit, sticky_bit, lsb_bit;
    logic [7:0] sub_shift;
    logic round_up;
    logic [23:0] rounded_mantissa;
    
    always_comb begin
        sub_shift = 8'd0;
        if (mult_res == 0) begin
            norm_exp = 0;
            shifted_frac = 0;
        end else if (exp_sum <= 0) begin 
            // Subnormal result
            sub_shift = 8'd1 - exp_sum[7:0];
            if (sub_shift > 47) sub_shift = 47;
            shifted_frac = {mult_res, 8'b0} >> sub_shift;
            norm_exp = 0;
        end else begin
            // Normal
            if (exp_sum - $signed({5'b0, lzc}) <= 0) begin
                sub_shift = 8'd1 - (exp_sum[7:0] - {3'b0, lzc});
                if (sub_shift > 47) sub_shift = 47;
                shifted_frac = { (mult_res << lzc), 8'b0 } >> sub_shift;
                norm_exp = 0;
            end else begin
                shifted_frac = { (mult_res << lzc), 8'b0 }; 
                norm_exp = exp_sum - $signed({5'b0, lzc});
            end
        end
        
        lsb_bit    = shifted_frac[32];     // LSB of 23-bit mantissa
        guard_bit  = shifted_frac[31];     // Guard bit
        round_bit  = shifted_frac[30];     // Round bit
        sticky_bit = |shifted_frac[29:0];  // Sticky bit
        
        // RTNE: round up if Guard & (Round | Sticky | LSB)
        round_up = guard_bit & (round_bit | sticky_bit | lsb_bit);
        
        rounded_mantissa = shifted_frac[54:32] + round_up;
        
        final_exp = norm_exp[7:0];
        final_mantissa = rounded_mantissa[22:0];
        
        if (rounded_mantissa[23]) begin 
            if (norm_exp == 0) begin
                final_exp = 1;
                final_mantissa = 0;
            end else begin
                final_exp = norm_exp[7:0] + 1;
                final_mantissa = 0;
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
        end else if ((a_isInf && b_isZero) || (a_isZero && b_isInf)) begin
            result = 32'h7FC00000; // NaN
        end else if (a_isInf || b_isInf) begin
            result = {sign_res, 31'h7F800000}; // Inf
        end else if (a_isZero || b_isZero) begin
            result = {sign_res, 31'h00000000}; // Zero
        end else if (norm_exp >= 10'sd255 || final_exp >= 8'hFF) begin
            result = {sign_res, 31'h7F800000}; // Overflow to Inf
        end else begin
            result = {sign_res, final_exp, final_mantissa};
        end
    end

endmodule
