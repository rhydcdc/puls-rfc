`timescale 1ns / 1ps

// =============================================================================
// Module: dequant_unit
// Project: Flash-SFU H9-O32 (FP8 KV Cache extension)
// Description: 128-lane FP8 E4M3 → FP16 dequant array.
//              SRAM byte-stream → MAC array input 사이에 inline 배치.
//              Per-lane: combinational (0 cycle); wrapper adds 1-cycle output
//              register optionally controlled by REG_OUT parameter to align
//              with downstream MAC pipeline timing.
//
// Inputs:
//   fp8_in   : 128 lanes of E4M3 (8 bit each)         — packed [127:0] × 8b
//   valid_in : data valid pulse
//   cfg_mode : 1'b0 = bypass (legacy FP16 path), 1'b1 = FP8 dequant active
//   fp16_pass: legacy FP16 vector (used when cfg_mode == 0)
//
// Outputs:
//   fp16_out : 128 lanes of FP16 (16 bit each)        — packed [127:0] × 16b
//   valid_out: aligned with fp16_out (matches REG_OUT latency)
//
// Pipeline Latency:
//   REG_OUT = 0 → combinational (0 cycle)
//   REG_OUT = 1 → 1 cycle (default, recommended for timing closure)
// =============================================================================

module dequant_unit #(
    parameter int VEC_DIM = 128,
    parameter bit REG_OUT = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    // Data inputs
    input  logic [7:0]        fp8_in    [0:VEC_DIM-1],   // FP8 byte stream
    input  logic [15:0]       fp16_pass [0:VEC_DIM-1],   // legacy FP16 path
    input  logic              valid_in,

    // Mode select
    input  logic              cfg_mode,                  // 0 = bypass, 1 = dequant

    // Outputs
    output logic [15:0]       fp16_out  [0:VEC_DIM-1],
    output logic              valid_out
);

    // -------------------------------------------------------------------------
    // 1. 128 parallel per-lane dequant (combinational)
    // -------------------------------------------------------------------------
    logic [15:0] lane_out [0:VEC_DIM-1];

    genvar i;
    generate
        for (i = 0; i < VEC_DIM; i = i + 1) begin : gen_lane
            fp8_e4m3_to_fp16_lane u_lane (
                .fp8_in   (fp8_in[i]),
                .fp16_out (lane_out[i])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 2. Mode mux: cfg_mode chooses dequant vs legacy FP16 pass-through
    // -------------------------------------------------------------------------
    logic [15:0] mux_out [0:VEC_DIM-1];

    always_comb begin
        for (int k = 0; k < VEC_DIM; k++) begin
            mux_out[k] = cfg_mode ? lane_out[k] : fp16_pass[k];
        end
    end

    // -------------------------------------------------------------------------
    // 3. Optional output register (REG_OUT = 1, default)
    // -------------------------------------------------------------------------
    generate
        if (REG_OUT) begin : gen_reg_out
            logic        valid_q;
            logic [15:0] out_q [0:VEC_DIM-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_q <= 1'b0;
                    for (int k = 0; k < VEC_DIM; k++) out_q[k] <= 16'h0000;
                end else begin
                    valid_q <= valid_in;
                    if (valid_in) begin
                        for (int k = 0; k < VEC_DIM; k++) out_q[k] <= mux_out[k];
                    end
                end
            end

            assign valid_out = valid_q;
            always_comb begin
                for (int k = 0; k < VEC_DIM; k++) fp16_out[k] = out_q[k];
            end
        end else begin : gen_comb
            assign valid_out = valid_in;
            always_comb begin
                for (int k = 0; k < VEC_DIM; k++) fp16_out[k] = mux_out[k];
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Trace points (synthesis-safe: translate_off/on)
    // -------------------------------------------------------------------------
    // synthesis translate_off
    always @(posedge clk) begin
        if (valid_out && cfg_mode)
            $display("[DEQUANT_OUT] time=%0t | fp8[0]=%h -> fp16[0]=%h | fp8[127]=%h -> fp16[127]=%h",
                     $time, fp8_in[0], fp16_out[0], fp8_in[127], fp16_out[127]);
    end
    // synthesis translate_on

endmodule
