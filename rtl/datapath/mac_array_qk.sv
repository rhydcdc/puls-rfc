`timescale 1ns / 1ps

// =============================================================================
// Module: mac_array_qk
// Project: Flash-SFU H9-O32
// Description: 128-lane Q·K FP16 dot-product → FP16 Score output.
//              Stg1:    128× fp16_to_fp32_mult (FP16×FP16→FP32)
//              Stg2a-1: Adder Tree L1 (128→64)
//              Stg2a-2: Adder Tree L2 (64→32)
//              Stg2b:   Adder Tree L3 (32→16)
//              Stg3a:   Adder Tree L4 (16→8)
//              Stg3b:   Adder Tree L5 (8→4)
//              Stg4a-1: Adder Tree L6 (4→2)
//              Stg4a-2: Adder Tree L7 (2→1)
//              Stg4b:   fp32_to_fp16_caster → s_out
// Pipeline Latency: 9 cycles (S3 full pipelining, all paths ≤0.7ns @7nm)
// =============================================================================

module mac_array_qk (
    input  logic              clk,
    input  logic              rst_n,

    // Inputs (128 parallel lanes)
    input  logic [15:0]       q_vec [0:127],  // Fixed Q vector (latched at row start)
    input  logic [15:0]       k_vec [0:127],  // Streaming K vector
    input  logic              valid_in,

    // Outputs
    output logic [15:0]       s_out,          // FP16 Score
    output logic              valid_out       // 9-cycle delayed valid
);

    // ========================================================================
    // Architecture Constants
    // ========================================================================
    localparam int VEC_DIM = 128;

    // ========================================================================
    // Internal Signals
    // ========================================================================

    // Pipeline Registers
    logic [31:0] stg1_reg       [0:127]; // After Multiplier (FP32)
    logic [31:0] stg2a_L1_reg  [0:63];  // ★ NEW: After L1 (128→64)
    logic [31:0] stg2_mid_reg  [0:31];  // After L2 (64→32)
    logic [31:0] stg2a_reg     [0:15];  // After L3 (32→16)
    logic [31:0] stg3_mid_reg  [0:7];   // ★ NEW: After L4 (16→8)
    logic [31:0] stg2b_reg     [0:3];   // After L5 (8→4)
    logic [31:0] stg4_L6_reg   [0:1];   // ★ NEW: After L6 (4→2)
    logic [31:0] stg4_L7_reg;           // ★ NEW: After L7 (2→1)

    // Combinational Wires
    logic [31:0] mult_out  [0:127]; // FP32 custom multiplier outputs
    logic [31:0] add_L1    [0:63];
    logic [31:0] add_L2    [0:31];
    logic [31:0] add_L3    [0:15];
    logic [31:0] add_L4    [0:7];
    logic [31:0] add_L5    [0:3];
    logic [31:0] add_L6    [0:1];
    logic [31:0] add_L7;

    // Valid shift register (9 cycles — S3 full pipelining)
    logic [8:0] valid_sr;

    // ========================================================================
    // Pipeline Valid Shift Register (9 Cycles)
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_sr <= 9'b0;
        end else begin
            valid_sr <= {valid_sr[7:0], valid_in};
        end
    end

    assign valid_out = valid_sr[8];

    // ========================================================================
    // Stage 1: 128 Parallel fp16_to_fp32_mult (Cycle 1)
    // ========================================================================
    genvar i;
    generate
        for (i = 0; i < VEC_DIM; i = i + 1) begin : gen_mult
            fp16_to_fp32_mult u_mult (
                .a      (q_vec[i]),
                .b      (k_vec[i]),
                .result (mult_out[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < VEC_DIM; j++) begin
                stg1_reg[j] <= 32'h0000_0000;
            end
        end else begin
            if (valid_in) begin
                for (int j = 0; j < VEC_DIM; j++) begin
                    stg1_reg[j] <= mult_out[j];
                end
            end
        end
    end

    // ========================================================================
    // Stage 2a-1: Adder Tree L1 (128→64) (Cycle 2)
    // ========================================================================
    generate
        for (i = 0; i < 64; i = i + 1) begin : gen_add_L1
            fp32_add u_add_L1 (
                .a      (stg1_reg[2*i]),
                .b      (stg1_reg[2*i+1]),
                .result (add_L1[i])
            );
        end
    endgenerate

    // ★ Pipeline register: stg2a_L1_reg (64 × 32-bit = 2,048 FF)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < 64; j++) begin
                stg2a_L1_reg[j] <= 32'h0000_0000;
            end
        end else begin
            if (valid_sr[0]) begin
                for (int j = 0; j < 64; j++) begin
                    stg2a_L1_reg[j] <= add_L1[j];
                end
            end
        end
    end

    // ========================================================================
    // Stage 2a-2: Adder Tree L2 (64→32) (Cycle 3)
    // ========================================================================
    generate
        for (i = 0; i < 32; i = i + 1) begin : gen_add_L2
            fp32_add u_add_L2 (
                .a      (stg2a_L1_reg[2*i]),
                .b      (stg2a_L1_reg[2*i+1]),
                .result (add_L2[i])
            );
        end
    endgenerate

    // Pipeline register: stg2_mid_reg (32 × 32-bit)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < 32; j++) begin
                stg2_mid_reg[j] <= 32'h0000_0000;
            end
        end else begin
            if (valid_sr[1]) begin
                for (int j = 0; j < 32; j++) begin
                    stg2_mid_reg[j] <= add_L2[j];
                end
            end
        end
    end

    // ========================================================================
    // Stage 2b: Adder Tree L3 (32→16) (Cycle 4)
    // ========================================================================
    generate
        for (i = 0; i < 16; i = i + 1) begin : gen_add_L3
            fp32_add u_add_L3 (
                .a      (stg2_mid_reg[2*i]),
                .b      (stg2_mid_reg[2*i+1]),
                .result (add_L3[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < 16; j++) begin
                stg2a_reg[j] <= 32'h0000_0000;
            end
        end else begin
            if (valid_sr[2]) begin
                for (int j = 0; j < 16; j++) begin
                    stg2a_reg[j] <= add_L3[j];
                end
            end
        end
    end

    // ========================================================================
    // Stage 3a: Adder Tree L4 (16→8) (Cycle 5)
    // ========================================================================
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_add_L4
            fp32_add u_add_L4 (
                .a      (stg2a_reg[2*i]),
                .b      (stg2a_reg[2*i+1]),
                .result (add_L4[i])
            );
        end
    endgenerate

    // ★ Pipeline register: stg3_mid_reg (8 × 32-bit = 256 FF)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < 8; j++) begin
                stg3_mid_reg[j] <= 32'h0000_0000;
            end
        end else begin
            if (valid_sr[3]) begin
                for (int j = 0; j < 8; j++) begin
                    stg3_mid_reg[j] <= add_L4[j];
                end
            end
        end
    end

    // ========================================================================
    // Stage 3b: Adder Tree L5 (8→4) (Cycle 6)
    // ========================================================================
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_add_L5
            fp32_add u_add_L5 (
                .a      (stg3_mid_reg[2*i]),
                .b      (stg3_mid_reg[2*i+1]),
                .result (add_L5[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < 4; j++) begin
                stg2b_reg[j] <= 32'h0000_0000;
            end
        end else begin
            if (valid_sr[4]) begin
                for (int j = 0; j < 4; j++) begin
                    stg2b_reg[j] <= add_L5[j];
                end
            end
        end
    end

    // ========================================================================
    // Stage 4a-1: Adder Tree L6 (4→2) (Cycle 7)
    // ========================================================================
    generate
        for (i = 0; i < 2; i = i + 1) begin : gen_add_L6
            fp32_add u_add_L6 (
                .a      (stg2b_reg[2*i]),
                .b      (stg2b_reg[2*i+1]),
                .result (add_L6[i])
            );
        end
    endgenerate

    // ★ Pipeline register: stg4_L6_reg (2 × 32-bit = 64 FF)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg4_L6_reg[0] <= 32'h0000_0000;
            stg4_L6_reg[1] <= 32'h0000_0000;
        end else begin
            if (valid_sr[5]) begin
                stg4_L6_reg[0] <= add_L6[0];
                stg4_L6_reg[1] <= add_L6[1];
            end
        end
    end

    // ========================================================================
    // Stage 4a-2: Adder Tree L7 (2→1) (Cycle 8)
    // ========================================================================
    fp32_add u_add_L7 (
        .a      (stg4_L6_reg[0]),
        .b      (stg4_L6_reg[1]),
        .result (add_L7)
    );

    // ★ Pipeline register: stg4_L7_reg (1 × 32-bit = 32 FF)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg4_L7_reg <= 32'h0000_0000;
        end else begin
            if (valid_sr[6]) begin
                stg4_L7_reg <= add_L7;
            end
        end
    end

    // ========================================================================
    // Stage 4b: FP32→FP16 Cast (Cycle 9)
    // ========================================================================
    logic [15:0] s_out_fp16;
    fp32_to_fp16_caster u_cast_final (
        .a (stg4_L7_reg),
        .q (s_out_fp16)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_out <= 16'h0000;
        end else begin
            if (valid_sr[7]) begin
                s_out <= s_out_fp16;
            end
        end
    end

    // ========================================================================
    // Trace Points — Synthesis-safe: translate_off/on
    // ========================================================================
    // synthesis translate_off
    always @(posedge clk) begin
        if (valid_in)
            $display("[MAC_QK_STG1_IN]  time=%0t | k_vec[0]=%h q_vec[0]=%h", $time, k_vec[0], q_vec[0]);
        if (valid_sr[0])
            $display("[MAC_QK_STG2a1]   time=%0t | stg1_reg[0]=%h stg1_reg[127]=%h | add_L1[0]=%h add_L1[63]=%h",
                     $time, stg1_reg[0], stg1_reg[127], add_L1[0], add_L1[63]);
        if (valid_sr[1])
            $display("[MAC_QK_STG2a2]   time=%0t | stg2a_L1_reg[0]=%h | add_L2[0]=%h add_L2[31]=%h",
                     $time, stg2a_L1_reg[0], add_L2[0], add_L2[31]);
        if (valid_sr[2])
            $display("[MAC_QK_STG2b]    time=%0t | stg2_mid_reg[0]=%h | add_L3[0]=%h",
                     $time, stg2_mid_reg[0], add_L3[0]);
        if (valid_sr[3])
            $display("[MAC_QK_STG3a]    time=%0t | stg2a_reg[0]=%h | add_L4[0]=%h",
                     $time, stg2a_reg[0], add_L4[0]);
        if (valid_sr[4])
            $display("[MAC_QK_STG3b]    time=%0t | stg3_mid_reg[0]=%h | add_L5[0]=%h",
                     $time, stg3_mid_reg[0], add_L5[0]);
        if (valid_sr[5])
            $display("[MAC_QK_STG4a1]   time=%0t | stg2b_reg[0]=%h | add_L6[0]=%h add_L6[1]=%h",
                     $time, stg2b_reg[0], add_L6[0], add_L6[1]);
        if (valid_sr[6])
            $display("[MAC_QK_STG4a2]   time=%0t | stg4_L6_reg[0]=%h | add_L7=%h",
                     $time, stg4_L6_reg[0], add_L7);
        if (valid_sr[7])
            $display("[MAC_QK_STG4b]    time=%0t | stg4_L7_reg=%h | s_out_fp16=%h",
                     $time, stg4_L7_reg, s_out_fp16);
        if (valid_out)
            $display("[MAC_QK_SOUT]     time=%0t | s_out=%h (latency=9)", $time, s_out);
    end
    // synthesis translate_on

endmodule
