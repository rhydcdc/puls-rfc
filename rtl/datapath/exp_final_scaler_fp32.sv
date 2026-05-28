`timescale 1ns / 1ps

// =============================================================================
// Module: exp_final_scaler_fp32
// Project: Flash-SFU H9-O32
// Description: Final scaler for EXP pipeline. Multiplies the FP32 fractional
//              result from Horner pipeline by 2^n_int.
// Pipeline Latency: 1 cycle
// =============================================================================

module exp_final_scaler_fp32 (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [31:0]        horner_out,   // FP32 from Horner Stage 9
    input  logic signed [4:0]  n_int,        // Integer part from Stage 2
    output logic [31:0]        scaler_out    // FP32 output = 2^n × horner_out
);

    logic signed [9:0] exp_original;
    logic signed [9:0] n_ext;
    logic signed [9:0] exp_sum;

    assign exp_original = {2'b00, horner_out[30:23]};
    assign n_ext = {{5{n_int[4]}}, n_int};        // 5bit → 10bit sign extension
    assign exp_sum = exp_original + n_ext;

    logic [31:0] scaler_out_comb;
    // Underflow -> 0.0
    // Overflow -> Inf (Max FP32)
    assign scaler_out_comb = (exp_sum <= 10'sd0)   ? 32'h00000000 : 
                             (exp_sum >= 10'sd255) ? {horner_out[31], 8'hFF, 23'h0} :
                             {horner_out[31], exp_sum[7:0], horner_out[22:0]};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaler_out <= 32'h0;
        end else begin
            scaler_out <= scaler_out_comb;
        end
    end

endmodule
