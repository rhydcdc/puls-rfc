`timescale 1ns / 1ps

// =============================================================================
// Module: mac_array_sv
// Project: Flash-SFU H9-O32
// Description: SV (Score-Value) MAC array with FP32 Accumulation, Shadow Register,
//              and 8-Chunk Write-Back multiplexer.
// Pipeline Latency: 2 cycles delay for valid_out
// =============================================================================

module mac_array_sv (
    input  logic              clk,
    input  logic              rst_n,

    // Data Inputs
    input  logic [31:0]       scalar_in,      // FP32 X_out or Y_j
    input  logic [31:0]       vec_in [0:127], // FP32: init=O_prev, normal=V
    
    // Control
    input  logic              valid_in,
    input  logic              init_mode,      // 1: O_reg <- scalar * vec (X*O_prev)
    input  logic              shadow_latch,   // Shadow Register 래치
    input  logic [2:0]        chunk_sel,      // 3bit 0~7 chunk select
    
    // Outputs
    output logic [511:0]      o_chunk_out,    // 16레인 FP32 chunk -> State RF
    output logic              valid_out       // 2사이클 지연
);

    // -------------------------------------------------------------------------
    // Stage 1: 128 Parallel FP32 Multiply (Cycle 1)
    // -------------------------------------------------------------------------
    logic [31:0] mult_out [0:127];
    logic [31:0] mult_reg [0:127];
    logic        valid_d1;
    logic        init_mode_d1;

    generate
        for (genvar i = 0; i < 128; i++) begin : gen_mult
            fp32_mult u_mult (
                .a(scalar_in),
                .b(vec_in[i]),
                .result(mult_out[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            init_mode_d1 <= 1'b0;
            for (int k = 0; k < 128; k++) begin
                mult_reg[k] <= 32'h0;
            end
        end else begin
            valid_d1 <= valid_in;
            init_mode_d1 <= init_mode;
            if (valid_in) begin
                for (int k = 0; k < 128; k++) begin
                    mult_reg[k] <= mult_out[k];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: 128 Parallel FP32 Accumulate (Cycle 2)
    // -------------------------------------------------------------------------
    logic [31:0] add_in_b [0:127];
    logic [31:0] add_out  [0:127];
    logic [31:0] O_reg    [0:127];
    logic        valid_d2;

    always_comb begin
        for (int k = 0; k < 128; k++) begin
            add_in_b[k] = init_mode_d1 ? 32'h0 : O_reg[k];
        end
    end

    generate
        for (genvar i = 0; i < 128; i++) begin : gen_add
            fp32_add u_add (
                .a(mult_reg[i]),
                .b(add_in_b[i]),
                .result(add_out[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d2 <= 1'b0;
            for (int k = 0; k < 128; k++) begin
                O_reg[k] <= 32'h0;
            end
        end else begin
            valid_d2 <= valid_d1;
            if (valid_d1) begin
                for (int k = 0; k < 128; k++) begin
                    O_reg[k] <= add_out[k];
                end
            end
        end
    end

    assign valid_out = valid_d2;

    // -------------------------------------------------------------------------
    // Shadow Register (FP32 래치, Caster 삭제)
    // -------------------------------------------------------------------------
    logic [31:0] shadow_reg [0:127];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int k = 0; k < 128; k++) begin
                shadow_reg[k] <= 32'h0;
            end
        end else begin
            if (shadow_latch) begin
                for (int k = 0; k < 128; k++) begin
                    shadow_reg[k] <= O_reg[k];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 8-Chunk Multiplexer (FP32 * 16 lanes = 512 bits per chunk)
    // -------------------------------------------------------------------------
    always_comb begin
        o_chunk_out = 512'b0;
        case (chunk_sel)
            3'd0: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[k];
            3'd1: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[16+k];
            3'd2: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[32+k];
            3'd3: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[48+k];
            3'd4: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[64+k];
            3'd5: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[80+k];
            3'd6: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[96+k];
            3'd7: for (int k = 0; k < 16; k++) o_chunk_out[k*32 +: 32] = shadow_reg[112+k];
            default: o_chunk_out = 512'b0;
        endcase
    end

endmodule
