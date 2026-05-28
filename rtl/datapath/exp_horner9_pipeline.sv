`timescale 1ns / 1ps

// =============================================================================
// Module: exp_horner9_pipeline
// Project: Flash-SFU H9-O32
// Description:  9th-order Horner polynomial evaluation in FP32 format for exp
//               approximation. Converts fractional part of FP16 input to FP32,
//               evaluates 2^f, and produces FP32 output in 18 cycles.
//               Each Horner stage split: fp32_mult → [FF] → fp32_add → [FF]
// Pipeline Latency: 18 cycles (S3 full pipelining, all paths ≤0.7ns @7nm)
// =============================================================================

module exp_horner9_pipeline (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] frac_fp16,     // f input (FP16, from exp_fp16_normalizer)
    input  logic        valid_in,      // Data valid signal
    output logic [31:0] pow2f_fp32,    // 2^f output (FP32)
    output logic        valid_out      // Output valid (18 cycles after valid_in)
);

    // =========================================================================
    // 1. FP32 Constants (Coefficients)
    // =========================================================================
    localparam logic [31:0] C0 = 32'h3F800000; // 1.0
    localparam logic [31:0] C1 = 32'h3F317218; // 0.6931471805599453
    localparam logic [31:0] C2 = 32'h3E75FDF0; // 0.2402265069591007
    localparam logic [31:0] C3 = 32'h3D635847; // 0.05550410866482157
    localparam logic [31:0] C4 = 32'h3C1D955B; // 0.009618129107628477
    localparam logic [31:0] C5 = 32'h3AAEC3FF; // 0.0013333558146428443
    localparam logic [31:0] C6 = 32'h39218489; // 0.00015403530393381407
    localparam logic [31:0] C7 = 32'h377FE5FE; // 0.000015252733804059458
    localparam logic [31:0] C8 = 32'h35B16011; // 0.0000013215486790144309
    localparam logic [31:0] C9 = 32'h33DA929F; // 0.00000010178086009239699

    // =========================================================================
    // 2. Input Conversion: FP16 -> FP32
    // =========================================================================
    logic [31:0] f_fp32;
    fp16_to_fp32_caster u_cast_input (
        .a(frac_fp16),
        .q(f_fp32)
    );

    // =========================================================================
    // 3. Shift Registers
    // =========================================================================
    // f propagation shift register (17 stages, FP32)
    // Stage N (N≥2) mult uses f_d[2*(N-1)-1] due to 2-cycle per stage
    logic [31:0] f_d [0:16];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 17; i++) begin
                f_d[i] <= 32'h0;
            end
        end else begin
            f_d[0] <= f_fp32;         // Stage 1 -> Stage 2 logic
            for (int i = 1; i < 17; i++) begin
                f_d[i] <= f_d[i-1];   // Stage N -> Stage N+1 logic
            end
        end
    end

    // Valid shift register (18 bits)
    logic [17:0] valid_sr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_sr <= 18'b0;
        end else begin
            valid_sr <= {valid_sr[16:0], valid_in};
        end
    end

    assign valid_out = valid_sr[17];

    // =========================================================================
    // 4. 9-Stage Horner MAC Pipeline (2 cycles per stage = 18 cycles total)
    // =========================================================================

    // -------------------------------------------------------------------------
    // Stage 1a: mult only — f_fp32 * C9
    // -------------------------------------------------------------------------
    logic [31:0] mult_s1, mult_r1;
    fp32_mult u_mult_s1 (.a(f_fp32),  .b(C9),      .result(mult_s1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r1 <= 32'h0;
        else        mult_r1 <= mult_s1;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 1b: add only — C8 + mult_r1
    // -------------------------------------------------------------------------
    logic [31:0] add_s1, acc_r1;
    fp32_add  u_add_s1  (.a(C8),      .b(mult_r1), .result(add_s1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r1 <= 32'h0;
        else        acc_r1 <= add_s1;
    end

    // -------------------------------------------------------------------------
    // Stage 2a: mult only — f_d[1] * acc_r1
    // -------------------------------------------------------------------------
    logic [31:0] mult_s2, mult_r2;
    fp32_mult u_mult_s2 (.a(f_d[1]),  .b(acc_r1),  .result(mult_s2));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r2 <= 32'h0;
        else        mult_r2 <= mult_s2;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 2b: add only — C7 + mult_r2
    // -------------------------------------------------------------------------
    logic [31:0] add_s2, acc_r2;
    fp32_add  u_add_s2  (.a(C7),      .b(mult_r2), .result(add_s2));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r2 <= 32'h0;
        else        acc_r2 <= add_s2;
    end

    // -------------------------------------------------------------------------
    // Stage 3a: mult only — f_d[3] * acc_r2
    // -------------------------------------------------------------------------
    logic [31:0] mult_s3, mult_r3;
    fp32_mult u_mult_s3 (.a(f_d[3]),  .b(acc_r2),  .result(mult_s3));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r3 <= 32'h0;
        else        mult_r3 <= mult_s3;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 3b: add only — C6 + mult_r3
    // -------------------------------------------------------------------------
    logic [31:0] add_s3, acc_r3;
    fp32_add  u_add_s3  (.a(C6),      .b(mult_r3), .result(add_s3));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r3 <= 32'h0;
        else        acc_r3 <= add_s3;
    end

    // -------------------------------------------------------------------------
    // Stage 4a: mult only — f_d[5] * acc_r3
    // -------------------------------------------------------------------------
    logic [31:0] mult_s4, mult_r4;
    fp32_mult u_mult_s4 (.a(f_d[5]),  .b(acc_r3),  .result(mult_s4));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r4 <= 32'h0;
        else        mult_r4 <= mult_s4;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 4b: add only — C5 + mult_r4
    // -------------------------------------------------------------------------
    logic [31:0] add_s4, acc_r4;
    fp32_add  u_add_s4  (.a(C5),      .b(mult_r4), .result(add_s4));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r4 <= 32'h0;
        else        acc_r4 <= add_s4;
    end

    // -------------------------------------------------------------------------
    // Stage 5a: mult only — f_d[7] * acc_r4
    // -------------------------------------------------------------------------
    logic [31:0] mult_s5, mult_r5;
    fp32_mult u_mult_s5 (.a(f_d[7]),  .b(acc_r4),  .result(mult_s5));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r5 <= 32'h0;
        else        mult_r5 <= mult_s5;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 5b: add only — C4 + mult_r5
    // -------------------------------------------------------------------------
    logic [31:0] add_s5, acc_r5;
    fp32_add  u_add_s5  (.a(C4),      .b(mult_r5), .result(add_s5));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r5 <= 32'h0;
        else        acc_r5 <= add_s5;
    end

    // -------------------------------------------------------------------------
    // Stage 6a: mult only — f_d[9] * acc_r5
    // -------------------------------------------------------------------------
    logic [31:0] mult_s6, mult_r6;
    fp32_mult u_mult_s6 (.a(f_d[9]),  .b(acc_r5),  .result(mult_s6));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r6 <= 32'h0;
        else        mult_r6 <= mult_s6;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 6b: add only — C3 + mult_r6
    // -------------------------------------------------------------------------
    logic [31:0] add_s6, acc_r6;
    fp32_add  u_add_s6  (.a(C3),      .b(mult_r6), .result(add_s6));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r6 <= 32'h0;
        else        acc_r6 <= add_s6;
    end

    // -------------------------------------------------------------------------
    // Stage 7a: mult only — f_d[11] * acc_r6
    // -------------------------------------------------------------------------
    logic [31:0] mult_s7, mult_r7;
    fp32_mult u_mult_s7 (.a(f_d[11]), .b(acc_r6),  .result(mult_s7));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r7 <= 32'h0;
        else        mult_r7 <= mult_s7;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 7b: add only — C2 + mult_r7
    // -------------------------------------------------------------------------
    logic [31:0] add_s7, acc_r7;
    fp32_add  u_add_s7  (.a(C2),      .b(mult_r7), .result(add_s7));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r7 <= 32'h0;
        else        acc_r7 <= add_s7;
    end

    // -------------------------------------------------------------------------
    // Stage 8a: mult only — f_d[13] * acc_r7
    // -------------------------------------------------------------------------
    logic [31:0] mult_s8, mult_r8;
    fp32_mult u_mult_s8 (.a(f_d[13]), .b(acc_r7),  .result(mult_s8));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r8 <= 32'h0;
        else        mult_r8 <= mult_s8;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 8b: add only — C1 + mult_r8
    // -------------------------------------------------------------------------
    logic [31:0] add_s8, acc_r8;
    fp32_add  u_add_s8  (.a(C1),      .b(mult_r8), .result(add_s8));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_r8 <= 32'h0;
        else        acc_r8 <= add_s8;
    end

    // -------------------------------------------------------------------------
    // Stage 9a: mult only — f_d[15] * acc_r8
    // -------------------------------------------------------------------------
    logic [31:0] mult_s9, mult_r9;
    fp32_mult u_mult_s9 (.a(f_d[15]), .b(acc_r8),  .result(mult_s9));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mult_r9 <= 32'h0;
        else        mult_r9 <= mult_s9;     // ★ NEW FF
    end

    // -------------------------------------------------------------------------
    // Stage 9b: add only — C0 + mult_r9 → pow2f_fp32
    // -------------------------------------------------------------------------
    logic [31:0] add_s9;
    fp32_add  u_add_s9  (.a(C0),      .b(mult_r9), .result(add_s9));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pow2f_fp32 <= 32'h0;
        else        pow2f_fp32 <= add_s9;
    end

    // =========================================================================
    // Trace Points — Synthesis-safe: translate_off/on
    // =========================================================================
    // synthesis translate_off
    always @(posedge clk) begin
        if (valid_sr[0])   $display("[HORNER_S1a]  time=%0t | mult_r1=%h",                            $time, mult_s1);
        if (valid_sr[1])   $display("[HORNER_S1b]  time=%0t | acc_r1=%h",                             $time, add_s1);
        if (valid_sr[2])   $display("[HORNER_S2a]  time=%0t | f_d[1]=%h | mult_r2=%h",                $time, f_d[1], mult_s2);
        if (valid_sr[3])   $display("[HORNER_S2b]  time=%0t | acc_r2=%h",                             $time, add_s2);
        if (valid_sr[4])   $display("[HORNER_S3a]  time=%0t | f_d[3]=%h | mult_r3=%h",                $time, f_d[3], mult_s3);
        if (valid_sr[5])   $display("[HORNER_S3b]  time=%0t | acc_r3=%h",                             $time, add_s3);
        if (valid_sr[6])   $display("[HORNER_S4a]  time=%0t | f_d[5]=%h | mult_r4=%h",                $time, f_d[5], mult_s4);
        if (valid_sr[7])   $display("[HORNER_S4b]  time=%0t | acc_r4=%h",                             $time, add_s4);
        if (valid_sr[8])   $display("[HORNER_S5a]  time=%0t | f_d[7]=%h | mult_r5=%h",                $time, f_d[7], mult_s5);
        if (valid_sr[9])   $display("[HORNER_S5b]  time=%0t | acc_r5=%h",                             $time, add_s5);
        if (valid_sr[10])  $display("[HORNER_S6a]  time=%0t | f_d[9]=%h | mult_r6=%h",                $time, f_d[9], mult_s6);
        if (valid_sr[11])  $display("[HORNER_S6b]  time=%0t | acc_r6=%h",                             $time, add_s6);
        if (valid_sr[12])  $display("[HORNER_S7a]  time=%0t | f_d[11]=%h | mult_r7=%h",               $time, f_d[11], mult_s7);
        if (valid_sr[13])  $display("[HORNER_S7b]  time=%0t | acc_r7=%h",                             $time, add_s7);
        if (valid_sr[14])  $display("[HORNER_S8a]  time=%0t | f_d[13]=%h | mult_r8=%h",               $time, f_d[13], mult_s8);
        if (valid_sr[15])  $display("[HORNER_S8b]  time=%0t | acc_r8=%h",                             $time, add_s8);
        if (valid_sr[16])  $display("[HORNER_S9a]  time=%0t | f_d[15]=%h | mult_r9=%h",               $time, f_d[15], mult_s9);
        if (valid_sr[17])  $display("[HORNER_S9b]  time=%0t | pow2f=%h",                              $time, add_s9);
    end
    // synthesis translate_on

endmodule
