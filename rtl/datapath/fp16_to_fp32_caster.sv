`timescale 1ns / 1ps

// =============================================================================
// Module: fp16_to_fp32_caster
// Project: Flash-SFU H9-O32
// Description: FP16 to FP32 converter with FTZ for subnormals
// Pipeline Latency: Combinational
// =============================================================================

module fp16_to_fp32_caster (
    input  logic [15:0] a,
    output logic [31:0] q
);

    // FP16 Components
    logic        sign_a;
    logic [4:0]  exp_a;
    logic [9:0]  frac_a;

    assign sign_a = a[15];
    assign exp_a  = a[14:10];
    assign frac_a = a[9:0];

    // FP32 Components
    logic [7:0]  exp_q;
    logic [22:0] frac_q;

    always_comb begin
        // Default assignment for FP32 components
        exp_q  = 8'd0;
        frac_q = 23'd0;

        // Condition Check
        if (exp_a == 5'h00) begin
            if (frac_a == 10'd0) begin
                // Case: Zero
                exp_q  = 8'd0;
                frac_q = 23'd0;
            end else begin
                // Case: Subnormal
                // Strategy: Flush-to-Zero (FTZ) mode
                exp_q  = 8'd0;
                frac_q = 23'd0;
            end
        end
        else if (exp_a == 5'h1F) begin
            // Case: Inf or NaN
            exp_q = 8'hFF;
            if (frac_a == 10'd0) begin
                // Infinity
                frac_q = 23'd0;
            end else begin
                // NaN -> Set to Quiet NaN
                frac_q = 23'h400000;
            end
        end
        else begin
            // Case: Normal Number
            // FP32 Exp = FP16 Exp - 15 (FP16 bias) + 127 (FP32 bias) = FP16 Exp + 112
            exp_q  = {3'b000, exp_a} + 8'd112; 
            // FP32 Frac = FP16 Frac padded with 13 zeros on the right
            frac_q = {frac_a, 13'd0};
        end
    end

    // Pack into FP32 Format
    assign q = {sign_a, exp_q, frac_q};

endmodule
