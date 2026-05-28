`timescale 1ns / 1ps

// =============================================================================
// Module: pingpong_sram
// Project: Flash-SFU H9-O32  (FP8 KV Cache extension copy)
// Description: Double-buffered K/V SRAM for Ping-Pong continuous operation.
//              Storage logic is unchanged from baseline. Byte-enable not added
//              because FP8 mode reuses the existing 2048-bit row at half-width
//              (see usage protocol below).
//
// Storage protocol per row (DWIDTH = 2048 b = 256 B):
//   FP16 mode (cfg_mode = 0):
//     row[15:0]   = lane 0  (FP16)
//     row[31:16]  = lane 1
//     ...
//     row[2047:2032] = lane 127      ← 128 × 16 b = 2048 b, full row used
//
//   FP8 mode (cfg_mode = 1):
//     row[7:0]    = lane 0  (E4M3 byte)
//     row[15:8]   = lane 1
//     ...
//     row[1023:1016] = lane 127      ← 128 × 8 b = 1024 b, lower half only
//     row[2047:1024] = unused / don't-care (DMA may write 0 to save energy)
//
//   The decode at the top level (flash_sfu_core_top) selects between the
//   FP16 lane slicing (i*16+:16) and the FP8 lane slicing (i*8+:8) based on
//   cfg_mode and feeds the dequant_unit accordingly. This SRAM module itself
//   is mode-agnostic.
//
// Pipeline Latency: 1 cycle (Synchronous Read) — unchanged
// =============================================================================

module pingpong_sram #(
    parameter DWIDTH = 2048,
    parameter DEPTH  = 32,
    parameter AWIDTH = 5
)(
    input  logic              clk,
    input  logic              rst_n,

    // Bank Select:
    // 0: SFU reads Bank A, DMA writes Bank B
    // 1: SFU reads Bank B, DMA writes Bank A
    input  logic              bank_sel,

    // ------------------------------------------------------------------------
    // External Loader (DMA) Write Interface
    // ------------------------------------------------------------------------
    input  logic              ext_k_we,
    input  logic [AWIDTH-1:0] ext_k_addr,
    input  logic [DWIDTH-1:0] ext_k_wdata,

    input  logic              ext_v_we,
    input  logic [AWIDTH-1:0] ext_v_addr,
    input  logic [DWIDTH-1:0] ext_v_wdata,

    // ------------------------------------------------------------------------
    // SFU Internal Read Interface
    // ------------------------------------------------------------------------
    input  logic              sfu_k_re,
    input  logic [AWIDTH-1:0] sfu_k_addr,
    output logic [DWIDTH-1:0] sfu_k_rdata,

    input  logic              sfu_v_re,
    input  logic [AWIDTH-1:0] sfu_v_addr,
    output logic [DWIDTH-1:0] sfu_v_rdata
);

    // ========================================================================
    // Internal Memory Arrays (Dual Bank)
    // ========================================================================
    // Bank A
    // * synthesis inference constraints can be added here if needed *
    logic [DWIDTH-1:0] k_mem_a [0:DEPTH-1];
    logic [DWIDTH-1:0] v_mem_a [0:DEPTH-1];

    // Bank B
    logic [DWIDTH-1:0] k_mem_b [0:DEPTH-1];
    logic [DWIDTH-1:0] v_mem_b [0:DEPTH-1];

    // Read Data Registers (for 1-cycle latency pipeline match)
    logic [DWIDTH-1:0] k_rdata_a, v_rdata_a;
    logic [DWIDTH-1:0] k_rdata_b, v_rdata_b;

    // ========================================================================
    // Bank A Logic
    // ========================================================================
    // Write routing: when bank_sel == 1, DMA writes to Bank A
    logic bank_a_k_we;
    logic bank_a_v_we;
    
    // Read routing: when bank_sel == 0, SFU reads from Bank A
    logic bank_a_k_re;
    logic bank_a_v_re;

    assign bank_a_k_we = (bank_sel == 1'b1) ? ext_k_we : 1'b0;
    assign bank_a_v_we = (bank_sel == 1'b1) ? ext_v_we : 1'b0;

    assign bank_a_k_re = (bank_sel == 1'b0) ? sfu_k_re : 1'b0;
    assign bank_a_v_re = (bank_sel == 1'b0) ? sfu_v_re : 1'b0;

    always_ff @(posedge clk) begin
        // Write Port A
        if (bank_a_k_we) k_mem_a[ext_k_addr] <= ext_k_wdata;
        if (bank_a_v_we) v_mem_a[ext_v_addr] <= ext_v_wdata;

        // Read Port A (Synchronous Read)
        if (bank_a_k_re) k_rdata_a <= k_mem_a[sfu_k_addr];
        if (bank_a_v_re) v_rdata_a <= v_mem_a[sfu_v_addr];
    end

    // ========================================================================
    // Bank B Logic
    // ========================================================================
    // Write routing: when bank_sel == 0, DMA writes to Bank B
    logic bank_b_k_we;
    logic bank_b_v_we;
    
    // Read routing: when bank_sel == 1, SFU reads from Bank B
    logic bank_b_k_re;
    logic bank_b_v_re;

    assign bank_b_k_we = (bank_sel == 1'b0) ? ext_k_we : 1'b0;
    assign bank_b_v_we = (bank_sel == 1'b0) ? ext_v_we : 1'b0;

    assign bank_b_k_re = (bank_sel == 1'b1) ? sfu_k_re : 1'b0;
    assign bank_b_v_re = (bank_sel == 1'b1) ? sfu_v_re : 1'b0;

    always_ff @(posedge clk) begin
        // Write Port B
        if (bank_b_k_we) k_mem_b[ext_k_addr] <= ext_k_wdata;
        if (bank_b_v_we) v_mem_b[ext_v_addr] <= ext_v_wdata;

        // Read Port B (Synchronous Read)
        if (bank_b_k_re) k_rdata_b <= k_mem_b[sfu_k_addr];
        if (bank_b_v_re) v_rdata_b <= v_mem_b[sfu_v_addr];
    end

    // ========================================================================
    // Output Multiplexer (SFU Read Data)
    // ========================================================================
    // Selects the read data stream corresponding to the active SFU bank.
    // FSM ensures bank_sel is static during active Tile read phases.
    assign sfu_k_rdata = (bank_sel == 1'b0) ? k_rdata_a : k_rdata_b;
    assign sfu_v_rdata = (bank_sel == 1'b0) ? v_rdata_a : v_rdata_b;

endmodule
