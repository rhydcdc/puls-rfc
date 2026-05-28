`timescale 1ns / 1ps

// =============================================================================
// Module: exp_top
// Project: Flash-SFU H9-O32
// Description: Top module for 22-cycle EXP pipeline.
//              Combines Preprocessor (2 cyc), Horner9 (18 cyc), and Final Scaler (2 cyc).
// Pipeline Latency: 22 cycles (S3 full pipelining, all paths ≤0.7ns @7nm)
// =============================================================================

module exp_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  logic [15:0] x_in,       // FP16 input
    output logic        valid_out,  // 22-cycle delay
    output logic [31:0] exp_out     // FP32 output (e^x)
);

    // ═══════════════════════════════════════════════════════════════
    // Special Value Bypass Logic (21-stage SR)
    // ═══════════════════════════════════════════════════════════════
    logic        is_nan_in, is_zero_in, is_ninf_in, is_pinf_in;
    assign is_nan_in  = ((x_in >> 10) & 5'h1F) == 5'h1F && (x_in & 10'h3FF) != 0;
    assign is_ninf_in = (x_in == 16'hFC00);
    assign is_pinf_in = ((x_in & 16'h7FFF) == 16'h7C00) && !(x_in & 16'h8000);
    assign is_zero_in = (x_in == 16'h0000 || x_in == 16'h8000);

    logic        bypass_en_in;
    logic [31:0] bypass_val_in;
    assign bypass_en_in  = is_nan_in | is_zero_in | is_ninf_in | is_pinf_in;
    assign bypass_val_in = is_nan_in  ? 32'h7FC00000 :
                           is_ninf_in ? 32'h00000000 :
                           is_pinf_in ? 32'h7F7FFFFF :
                           is_zero_in ? 32'h3F800000 : 32'h00000000;

    logic bypass_en_sr [0:20];
    logic [31:0] bypass_val_sr [0:20];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<21; i++) begin
                bypass_en_sr[i]  <= 1'b0;
                bypass_val_sr[i] <= 32'h0;
            end
        end else begin
            bypass_en_sr[0]  <= bypass_en_in;
            bypass_val_sr[0] <= bypass_val_in;
            for (int i=1; i<21; i++) begin
                bypass_en_sr[i]  <= bypass_en_sr[i-1];
                bypass_val_sr[i] <= bypass_val_sr[i-1];
            end
        end
    end

    // ═══════════════════════════════════════════════════════════════
    // Stage 1: Base-2 Converter (1 cycle)
    // ═══════════════════════════════════════════════════════════════
    logic        valid_stg1;
    logic [15:0] y_stg1;

    exp_base2_converter u_base2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .x_in(x_in),
        .valid_out(valid_stg1),
        .y_out(y_stg1)
    );

    // ═══════════════════════════════════════════════════════════════
    // Stage 2: Int/Frac Separator + Normalizer (Combinational) + FF
    // ═══════════════════════════════════════════════════════════════
    logic signed [4:0] n_int_comb;
    logic [9:0]        f_fixed_comb;
    logic [15:0]       frac_fp16_comb;

    exp_int_frac_separator u_separator (
        .x_fp16(y_stg1),
        .int_part(n_int_comb),
        .f_fixed(f_fixed_comb)
    );

    exp_fp16_normalizer u_normalizer (
        .f_fixed(f_fixed_comb),
        .frac_fp16(frac_fp16_comb)
    );

    // Stage 2 Pipeline Register
    logic        valid_stg2;
    logic signed [4:0] n_int_stg2;
    logic [15:0]       frac_fp16_stg2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_stg2     <= 1'b0;
            n_int_stg2     <= 5'sd0;
            frac_fp16_stg2 <= 16'h0000;
        end else begin
            valid_stg2     <= valid_stg1;
            n_int_stg2     <= n_int_comb;
            frac_fp16_stg2 <= frac_fp16_comb;
        end
    end

    // ═══════════════════════════════════════════════════════════════
    // Stage 3~20: Horner 9차 FP32 MAC Pipeline (18 cycles)
    // ═══════════════════════════════════════════════════════════════
    logic        valid_horner_out;
    logic [31:0] horner_out;

    exp_horner9_pipeline u_horner (
        .clk(clk),
        .rst_n(rst_n),
        .frac_fp16(frac_fp16_stg2),
        .valid_in(valid_stg2),
        .pow2f_fp32(horner_out),
        .valid_out(valid_horner_out)
    );

    // ═══════════════════════════════════════════════════════════════
    // n_int 동기화 시프트 레지스터 (18단 — Horner 레이턴시 매칭)
    // ═══════════════════════════════════════════════════════════════
    logic signed [4:0] n_int_sr [0:17];  // 18단

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 18; i++)
                n_int_sr[i] <= 5'sd0;
        end else begin
            n_int_sr[0] <= n_int_stg2;
            for (int i = 1; i < 18; i++)
                n_int_sr[i] <= n_int_sr[i-1];
        end
    end

    // ═══════════════════════════════════════════════════════════════
    // Stage 21: Final Scaler (1 cycle) — FP32
    // ═══════════════════════════════════════════════════════════════
    logic [31:0] scaler_out_stg12;

    exp_final_scaler_fp32 u_scaler (
        .clk(clk),
        .rst_n(rst_n),
        .horner_out(horner_out),
        .n_int(n_int_sr[17]), // Sync with horner_out
        .scaler_out(scaler_out_stg12)
    );

    // valid_stg12 indicates the scaler output is valid
    logic valid_stg12;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_stg12 <= 1'b0;
        else        valid_stg12 <= valid_horner_out;
    end

    // ═══════════════════════════════════════════════════════════════
    // Stage 22: Final Output Register
    // ═══════════════════════════════════════════════════════════════
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            exp_out   <= 32'h00000000;
        end else begin
            valid_out <= valid_stg12;
            if (valid_stg12) begin
                if (bypass_en_sr[20]) begin
                    exp_out <= bypass_val_sr[20]; // Bypass path for special FP16 values
                end else begin
                    exp_out <= scaler_out_stg12;
                end
            end
        end
    end

endmodule
