`timescale 1ns / 1ps

// =============================================================================
// Module: q_broadcast_unit
// Project: Flash-SFU H9-O32
// Description: Stores 8 rows of Q vectors and broadcasts one based on row_cnt.
// Pipeline Latency: 1 cycle (from q_update pulse to q_vec_out output)
// =============================================================================

module q_broadcast_unit #(
    parameter USE_CDC = 0   // 0: Same clock domain (bypass CDC), 1: Enable 2-stage synchronizer
)(
    input  logic              clk,
    input  logic              rst_n,

    // ---- Q Reception Interface (from external Q Router) ----
    input  logic [2047:0]     q_data_in,      // 128 × FP16 = 2048-bit Q vector
    input  logic              q_valid_in,      // Handshake: data valid
    input  logic [2:0]        q_row_id_in,     // GQA slot index (0~7)
    output logic              q_ready_out,     // Handshake: ready to accept

    // ---- FSM Timing Interface ----
    input  logic [2:0]        row_cnt,         // Current processing row from main_fsm
    input  logic              q_update,        // Q latch trigger from main_fsm (pc=1)
    input  logic              core_busy,       // == top_running (write blocking during operation)

    // ---- Output to Datapath ----
    output logic [2047:0]     q_vec_out        // Direct connection to MAC_QK & exception_handler
);

    // ========================================================================
    // Block A: Optional CDC Synchronizer (2-Stage)
    // ========================================================================
    logic [2047:0]  q_data_sync;
    logic           q_valid_sync;
    logic [2:0]     q_row_id_sync;

    generate
        if (USE_CDC) begin : gen_cdc
            // 2-stage synchronizer for metastability protection
            logic           q_valid_meta;
            logic [2047:0]  q_data_meta;
            logic [2:0]     q_row_id_meta;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    q_valid_meta   <= 1'b0;
                    q_valid_sync   <= 1'b0;
                    q_data_meta    <= 2048'h0;
                    q_data_sync    <= 2048'h0;
                    q_row_id_meta  <= 3'd0;
                    q_row_id_sync  <= 3'd0;
                end else begin
                    // Stage 1
                    q_valid_meta   <= q_valid_in;
                    q_data_meta    <= q_data_in;
                    q_row_id_meta  <= q_row_id_in;
                    // Stage 2
                    q_valid_sync   <= q_valid_meta;
                    q_data_sync    <= q_data_meta;
                    q_row_id_sync  <= q_row_id_meta;
                end
            end
        end else begin : gen_no_cdc
            // Same clock domain: direct pass-through
            assign q_valid_sync  = q_valid_in;
            assign q_data_sync   = q_data_in;
            assign q_row_id_sync = q_row_id_in;
        end
    endgenerate

    // ========================================================================
    // Block B: Q Register File (8-Depth × 2048-bit)
    // ========================================================================
    logic [2047:0] q_rf [0:7];

    // Backpressure: accept writes only when core is not busy
    assign q_ready_out = ~core_busy;

    // Write handshake
    wire wr_handshake = q_valid_sync && q_ready_out;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < 8; j++)
                q_rf[j] <= 2048'h0;
        end else if (wr_handshake) begin
            q_rf[q_row_id_sync] <= q_data_sync;
        end
    end

    // ========================================================================
    // Block C: Output MUX & Latch Register
    // ========================================================================
    logic [2047:0] q_out_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_out_reg <= 2048'h0;
        end else if (q_update) begin
            // Latch the Q vector for the current row on q_update pulse (pc=1)
            // NBA ensures q_out_reg is stable from the next cycle onward,
            // perfectly aligned with K data arrival (SRAM 1-cycle latency at pc=2)
            q_out_reg <= q_rf[row_cnt];
        end
    end

    assign q_vec_out = q_out_reg;

endmodule
