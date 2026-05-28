`timescale 1ns / 1ps

// =============================================================================
// Module: fp32_to_fp16_caster
// Project: Flash-SFU H9-O32
// Description: FP32 to FP16 converter with RTNE rounding, FTZ, overflow saturation
// Pipeline Latency: Combinational
// =============================================================================

module fp32_to_fp16_caster (
    input  logic [31:0] a,
    output logic [15:0] q
);

    // FP32 Components
    logic        sign_a;
    logic [7:0]  exp_a;
    logic [22:0] frac_a;

    assign sign_a = a[31];
    assign exp_a  = a[30:23];
    assign frac_a = a[22:0];

    // FP16 Components
    logic [4:0]  exp_q;
    logic [9:0]  frac_q;

    // Intermediate variables for RTNE
    logic [7:0]  adjusted_exp;
    logic [10:0] rounded_frac; // 10 bits + 1 overflow bit
    logic        guard, round, sticky;
    logic        round_up;

    always_comb begin
        // Default assignments
        exp_q  = 5'd0;
        frac_q = 10'd0;
        adjusted_exp = 8'd0;
        rounded_frac = 11'd0;
        guard = 1'b0;
        round = 1'b0;
        sticky = 1'b0;
        round_up = 1'b0;

        // Condition Check
        if (exp_a <= 8'd112) begin
            // Underflow / Zero / Subnormal condition
            // FTZ (Flush-to-Zero) policy applied
            exp_q  = 5'd0;
            frac_q = 10'd0;
        end
        else if (exp_a == 8'hFF) begin
            // Infinity or NaN
            exp_q = 5'h1F;
            if (frac_a == 23'd0) begin
                // Infinity
                frac_q = 10'd0;
            end else begin
                // NaN -> Set to Quiet NaN in FP16 (Set highest fraction bit to 1)
                frac_q = 10'h200;
            end
        end
        else if (exp_a >= 8'd143) begin
            // Overflow: Exponent is too large to fit in FP16
            // Saturate to Infinity
            exp_q  = 5'h1F;
            frac_q = 10'd0;
        end
        else begin
            // Normal range numbers (113 <= exp_a <= 142)
            // Initial Exponent calculation: Target exponent is exp_a - 112
            adjusted_exp = exp_a - 8'd112;

            // Round-to-Nearest-Even (RTNE)
            // frac_a[22:13] -> Top 10 bits of fraction used for FP16
            // frac_a[12]    -> Guard
            // frac_a[11]    -> Round
            // frac_a[10:0]  -> Sticky
            guard  = frac_a[12];
            round  = frac_a[11];
            sticky = |frac_a[10:0];

            // Tie-to-Even algorithm:
            // Round up if G & (R | S | frac_a[13] (odd))
            round_up = guard & (round | sticky | frac_a[13]);

            rounded_frac = frac_a[22:13] + round_up;

            // Check if fraction overflowed (e.g. 1111111111 + 1 -> 10000000000)
            if (rounded_frac[10]) begin
                // Fraction overflowed. Increment exponent and clear fraction
                if (adjusted_exp == 8'd30) begin
                    // If exponent was max normal value, incrementing overflows to Infinity
                    exp_q  = 5'h1F;
                    frac_q = 10'd0;
                end else begin
                    exp_q  = adjusted_exp[4:0] + 5'd1;
                    frac_q = 10'd0;
                end
            end else begin
                // Normal rounded value
                exp_q  = adjusted_exp[4:0];
                frac_q = rounded_frac[9:0];
            end
        end
    end

    // Pack into FP16 Format
    assign q = {sign_a, exp_q, frac_q};

endmodule
