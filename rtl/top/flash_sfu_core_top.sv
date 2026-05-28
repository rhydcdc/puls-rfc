`timescale 1ns / 1ps

// =============================================================================
// Module : flash_sfu_core_top
// Project: H9-O32 Flash-SFU  (FP8 KV Cache extension copy)
// Description:
//   Top-level integration of all Flash-SFU sub-modules.
//   Key changes from legacy flash_sfu_top:
//     - 347-cycle timing (S3: QK 9-cycle + Horner 18-cycle, was 333)
//     - V vector FP16→FP32 casting via 128× fp16_to_fp32_caster
//     - O_prev assembly register: 4096-bit / 8-chunk (was 2048-bit / 4-chunk)
//     - MAC SV MUX: fully FP32 (scalar_in [31:0], vec_in [31:0]×128)
//     - chunk_sel / o_prev_chunk_sel: 3-bit (was 2-bit)
//     - l_prev snapshot latch (start_row timing)
//     - x_out / y_out from reduction_unit are FP32 (was FP16)
//
//   FP8 KV Cache extension (this copy):
//     - cfg_mode input port (1 bit): 0 = FP16 baseline, 1 = FP8 E4M3
//     - 128-lane dequant_unit inserted between SRAM read and MAC array
//       on both K and V paths (REG_OUT = 0, combinational, 0-cycle latency
//       so that all downstream pipeline timing is unchanged).
//     - dequant_unit's internal cfg_mode mux handles bypass vs dequant.
//     - FP16 lane slicing (i*16+:16) and FP8 lane slicing (i*8+:8) are both
//       wired to dequant_unit; the mux selects according to cfg_mode.
//     - SRAM module unchanged (storage protocol documented in pingpong_sram.sv).
// Pipeline Latency: N/A (top-level integration wrapper)
// =============================================================================

module flash_sfu_core_top #(
    parameter MAX_TILES = 16,
    parameter DWIDTH    = 2048,
    parameter AWIDTH    = 5
)(
    input  logic              clk,
    input  logic              rst_n,

    // ---- Host Interface ----
    input  logic              start,
    input  logic [7:0]        num_tiles,
    output logic              tile_ready,

    // ---- Q Broadcast Reception Interface ----
    input  logic [2047:0]     q_data_in,
    input  logic              q_valid_in,
    input  logic [2:0]        q_row_id_in,
    output logic              q_ready_out,

    // ---- External K/V SRAM Write (DMA) ----
    input  logic              ext_k_we, ext_v_we,
    input  logic [AWIDTH-1:0] ext_k_addr, ext_v_addr,
    input  logic [DWIDTH-1:0] ext_k_wdata, ext_v_wdata,

    // ---- Diagnostic Read Ports ----
    output logic [15:0]       m_prev_out,
    output logic [31:0]       l_prev_out,
    output logic [511:0]      o_chunk_read_out,   // FP32 chunk
    input  logic [2:0]        rd_chunk_sel,         // 3-bit

    // ---- FP8 KV Cache Mode Select ----
    // 0 = FP16 baseline (legacy path), 1 = FP8 E4M3 dequant on K/V
    input  logic              cfg_mode
);

    // ========================================================================
    // 1. Internal Wire Declarations
    // ========================================================================

    // -- FSM outputs --
    logic        bank_sel;
    logic        start_row;
    logic        q_update;
    logic        s_valid_in;
    logic        fifo_rd_en;
    logic        exp_x_en;
    logic        exp_y_en;
    logic        acc_init;
    logic        we_m;
    logic        we_l;
    logic        shadow_latch;
    logic        we_o;
    logic        o_prev_ld_en;
    logic [2:0]  o_prev_chunk_sel;   // 3-bit
    logic [2:0]  row_cnt;
    logic [2:0]  chunk_sel;          // 3-bit
    logic [2:0]  o_wr_row;           // ★ S3: FSM-latched O write row

    // -- SRAM internal read interface --
    logic [DWIDTH-1:0] sfu_k_rdata;
    logic [DWIDTH-1:0] sfu_v_rdata;
    logic              sfu_k_re;
    logic              sfu_v_re;
    logic [AWIDTH-1:0] sfu_k_addr;
    logic [AWIDTH-1:0] sfu_v_addr;

    // -- Q Broadcast output --
    logic [2047:0] q_vec_selected;

    // -- Unpacked vectors --
    logic [15:0] q_vec [0:127];
    logic [15:0] k_vec [0:127];

    // -- V vector FP16 → FP32 --
    logic [15:0] v_vec_fp16 [0:127];
    logic [31:0] v_vec_fp32 [0:127];

    // -- MAC QK outputs --
    logic [15:0] s_out_qk;
    logic        valid_out_qk;

    // -- Score FIFO --
    logic [15:0] s_out_fifo;
    logic        fifo_full;
    logic        fifo_empty;

    // -- Reduction Unit outputs (★FP32★) --
    logic [15:0] m_new_out;
    logic [31:0] l_new_out;
    logic [31:0] x_out;          // FP32
    logic [31:0] y_out;          // FP32
    logic [31:0] val_a_out;
    logic        x_out_valid;
    logic        y_out_valid;

    // -- MAC SV --
    logic [31:0] sv_scalar_in;          // FP32
    logic [31:0] sv_vec_in [0:127];     // FP32
    logic        sv_init_mode;
    logic [511:0] o_chunk_out;
    logic         sv_valid_out;

    // -- State RF --
    logic [15:0]  rf_m_prev;
    logic [31:0]  rf_l_prev;
    logic [511:0] o_prev_chunk_internal;

    // ========================================================================
    // 2. Top-Level Cycle Tracker (Synced with main_fsm for address generation)
    // ========================================================================
    logic [8:0] top_global_cycle;
    logic       top_running;
    logic       top_tile_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            top_running      <= 1'b0;
            top_global_cycle <= 9'd0;
            top_tile_done    <= 1'b0;
        end else begin
            if (start && !top_running) begin
                // First start or restart from ALL_DONE
                top_running      <= 1'b1;
                top_global_cycle <= 9'd0;
                top_tile_done    <= 1'b0;
            end else if (top_running) begin
                if (!top_tile_done) begin
                    if (top_global_cycle < 9'd346) begin
                        top_global_cycle <= top_global_cycle + 9'd1;
                    end else begin
                        top_tile_done <= 1'b1;
                    end
                end else begin
                    // Tile-done handshake: check if more tiles remain
                    top_tile_done <= 1'b0;
                    if (tile_ready) begin
                        // All done
                        top_running <= 1'b0;
                    end else begin
                        // Next tile
                        top_global_cycle <= 9'd0;
                    end
                end
            end else if (start) begin
                // Restart from ALL_DONE
                top_running      <= 1'b1;
                top_global_cycle <= 9'd0;
                top_tile_done    <= 1'b0;
            end
        end
    end

    // ========================================================================
    // 3. K SRAM Read Address Generator
    // ========================================================================
    // K is read from Cy0 (gc=0) to end of Row 7 (gc~271).
    // Within each 34-cycle Row period, K SRAM reads occur for 32 cycles (k_counter 0~31).
    // k_counter wraps every 34 cycles to match the Row period.
    logic [5:0] k_counter;    // 0~33 cycle within Row
    logic       k_active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_counter <= 6'd0;
            k_active  <= 1'b0;
        end else if (top_running && !top_tile_done) begin
            if (top_global_cycle == 9'd0 && !k_active) begin
                k_active  <= 1'b1;
                k_counter <= 6'd0;
            end else if (k_active) begin
                if (top_global_cycle > 9'd271) begin
                    k_active  <= 1'b0;
                end else if (k_counter == 6'd33) begin
                    k_counter <= 6'd0;
                end else begin
                    k_counter <= k_counter + 6'd1;
                end
            end
        end else begin
            k_active  <= 1'b0;
            k_counter <= 6'd0;
        end
    end

    assign sfu_k_re   = k_active && (k_counter < 6'd32);
    assign sfu_k_addr = k_counter[4:0];

    // ========================================================================
    // 4. V SRAM Read Address Generator
    // ========================================================================
    // ★ S3: V read starts at gc=66 to align V[0] data with Y[0] arrival at gc=67.
    //   acc_init at gc=66 (pc=32): MAC_SV init cycle (X * O_prev).  [+14 from S0]
    //   Y[0] first arrives at gc=67 (exp_y_en at gc=45, +22 EXP latency).
    //   V SRAM 1-cycle read latency → v_re at gc=66 gives V[0] rdata at gc=67.
    //   sv_v_valid (delayed sfu_v_re) = 1 at gc=67 → MAC_SV: Y[0] × V[0]. ✓
    // gc upper bound: Row 7 last V read at gc=336 (gc=305+31).
    // Conservative: v active ends at gc > 341.  [327+14=341]
    logic [5:0] v_counter;    // 0~33 cycle within V-Row
    logic       v_active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_counter <= 6'd0;
            v_active  <= 1'b0;
        end else if (top_running && !top_tile_done) begin
            if (top_global_cycle == 9'd65 && !v_active) begin
                v_active  <= 1'b1;
                v_counter <= 6'd0;
            end else if (v_active) begin
                if (top_global_cycle > 9'd341) begin
                    v_active  <= 1'b0;
                    v_counter <= 6'd0;
                end else if (v_counter == 6'd33) begin
                    v_counter <= 6'd0;
                end else begin
                    v_counter <= v_counter + 6'd1;
                end
            end
        end else begin
            v_active  <= 1'b0;
            v_counter <= 6'd0;
        end
    end

    assign sfu_v_re   = v_active && (v_counter < 6'd32);
    assign sfu_v_addr = v_counter[4:0];

    // ========================================================================
    // 5. Delayed Valid Signals (SRAM 1-cycle read latency alignment)
    // ========================================================================
    logic qk_valid_in;
    logic sv_v_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qk_valid_in <= 1'b0;
            sv_v_valid  <= 1'b0;
        end else begin
            qk_valid_in <= sfu_k_re;
            sv_v_valid  <= sfu_v_re;
        end
    end

    // ========================================================================
    // 6. Data Unpacking (Q: FP16 always; K/V: FP16 or FP8 selected by cfg_mode)
    // ========================================================================
    // Q is unaffected by FP8 KV mode (Q is broadcast already as FP16).
    generate
        for (genvar i = 0; i < 128; i++) begin : gen_q_unpack
            assign q_vec[i] = q_vec_selected[i*16 +: 16];
        end
    endgenerate

    // K/V dual-slice (FP8 byte and FP16 word from the same SRAM row)
    logic [7:0]  k_fp8_lane  [0:127];
    logic [15:0] k_fp16_lane [0:127];
    logic [7:0]  v_fp8_lane  [0:127];
    logic [15:0] v_fp16_lane [0:127];

    generate
        for (genvar i = 0; i < 128; i++) begin : gen_kv_slice
            // FP8 mode uses lower 1024 b of the 2048-b row (one byte per lane)
            assign k_fp8_lane[i]  = sfu_k_rdata[i*8  +: 8 ];
            assign v_fp8_lane[i]  = sfu_v_rdata[i*8  +: 8 ];
            // FP16 mode uses the full 2048-b row (one half-word per lane)
            assign k_fp16_lane[i] = sfu_k_rdata[i*16 +: 16];
            assign v_fp16_lane[i] = sfu_v_rdata[i*16 +: 16];
        end
    endgenerate

    // ---- 6a. K dequant_unit (combinational, 0-cycle latency) ----
    // cfg_mode = 0 → fp16_pass passes through unchanged (legacy FP16 path)
    // cfg_mode = 1 → fp8_in is dequantized to FP16 per OCP §5.1
    logic dequant_k_valid_unused;
    dequant_unit #(
        .VEC_DIM (128),
        .REG_OUT (1'b0)
    ) u_dequant_k (
        .clk       (clk),
        .rst_n     (rst_n),
        .fp8_in    (k_fp8_lane),
        .fp16_pass (k_fp16_lane),
        .valid_in  (sfu_k_re),
        .cfg_mode  (cfg_mode),
        .fp16_out  (k_vec),
        .valid_out (dequant_k_valid_unused)
    );

    // ---- 6b. V dequant_unit (combinational, 0-cycle latency) ----
    logic dequant_v_valid_unused;
    dequant_unit #(
        .VEC_DIM (128),
        .REG_OUT (1'b0)
    ) u_dequant_v (
        .clk       (clk),
        .rst_n     (rst_n),
        .fp8_in    (v_fp8_lane),
        .fp16_pass (v_fp16_lane),
        .valid_in  (sfu_v_re),
        .cfg_mode  (cfg_mode),
        .fp16_out  (v_vec_fp16),
        .valid_out (dequant_v_valid_unused)
    );

    // ★ V FP16 → FP32 Casting (128 instances) — unchanged ★
    generate
        for (genvar i = 0; i < 128; i++) begin : gen_v_cast
            fp16_to_fp32_caster u_v_cast (
                .a(v_vec_fp16[i]),
                .q(v_vec_fp32[i])
            );
        end
    endgenerate

    // ========================================================================
    // 7. Sub-Module Instantiations
    // ========================================================================

    // ---- 7.1 Main FSM ----
    main_fsm #(
        .MAX_TILES(MAX_TILES)
    ) u_main_fsm (
        .clk              (clk),
        .rst_n            (rst_n),
        .fifo_empty       (fifo_empty),    // score_fifo.empty → main_fsm
        .start            (start),
        .num_tiles        (num_tiles),
        .tile_ready       (tile_ready),
        .bank_sel         (bank_sel),
        .start_row        (start_row),
        .s_valid_in       (s_valid_in),
        .fifo_rd_en       (fifo_rd_en),
        .exp_x_en         (exp_x_en),
        .exp_y_en         (exp_y_en),
        .acc_init         (acc_init),
        .we_m             (we_m),
        .we_l             (we_l),
        .shadow_latch     (shadow_latch),
        .we_o             (we_o),
        .q_update         (q_update),
        .o_prev_ld_en     (o_prev_ld_en),
        .o_prev_chunk_sel (o_prev_chunk_sel),
        .row_cnt_out      (row_cnt),
        .chunk_sel        (chunk_sel),
        .o_wr_row         (o_wr_row)
    );

    // ---- 7.2 Q Broadcast Unit ----
    q_broadcast_unit #(
        .USE_CDC(0)
    ) u_q_broadcast (
        .clk          (clk),
        .rst_n        (rst_n),
        .q_data_in    (q_data_in),
        .q_valid_in   (q_valid_in),
        .q_row_id_in  (q_row_id_in),
        .q_ready_out  (q_ready_out),
        .row_cnt      (row_cnt),
        .q_update     (q_update),
        .core_busy    (top_running),
        .q_vec_out    (q_vec_selected)
    );

    // ---- 7.3 Ping-Pong SRAM ----
    pingpong_sram #(
        .DWIDTH(DWIDTH),
        .DEPTH (32),
        .AWIDTH(AWIDTH)
    ) u_pingpong_sram (
        .clk          (clk),
        .rst_n        (rst_n),
        .bank_sel     (bank_sel),
        .ext_k_we     (ext_k_we),
        .ext_k_addr   (ext_k_addr),
        .ext_k_wdata  (ext_k_wdata),
        .ext_v_we     (ext_v_we),
        .ext_v_addr   (ext_v_addr),
        .ext_v_wdata  (ext_v_wdata),
        .sfu_k_re     (sfu_k_re),
        .sfu_k_addr   (sfu_k_addr),
        .sfu_k_rdata  (sfu_k_rdata),
        .sfu_v_re     (sfu_v_re),
        .sfu_v_addr   (sfu_v_addr),
        .sfu_v_rdata  (sfu_v_rdata)
    );

    // ---- 7.4 MAC Array QK ----
    mac_array_qk u_mac_array_qk (
        .clk       (clk),
        .rst_n     (rst_n),
        .q_vec     (q_vec),
        .k_vec     (k_vec),
        .valid_in  (qk_valid_in),
        .s_out     (s_out_qk),
        .valid_out (valid_out_qk)
    );

    // ---- 7.5 Score FIFO ----
    score_fifo u_score_fifo (
        .clk        (clk),
        .rst_n      (rst_n),
        .fifo_flush (1'b0),          // No flush in normal operation
        .wr_en      (valid_out_qk),
        .wr_data    (s_out_qk),
        .rd_en      (fifo_rd_en),
        .rd_data    (s_out_fifo),
        .full       (fifo_full),
        .empty      (fifo_empty),
        .count      ()               // Unused at top level
    );

    // ---- 7.6 Reduction Unit ----
    // L_prev snapshot latch (start_row timing)
    logic [31:0] l_prev_ld_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            l_prev_ld_reg <= 32'h0;
        else if (start_row)
            l_prev_ld_reg <= rf_l_prev;
    end

    reduction_unit u_reduction_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .start_row    (start_row),
        .we_m         (we_m),             // ★ Phase5-3: pc=9 running_max reset trigger ★
        .s_valid_in   (s_valid_in),
        .exp_x_en     (exp_x_en),
        .exp_y_en     (exp_y_en),
        .acc_init     (acc_init),
        .s_in         (s_out_qk),
        .s_out_q      (s_out_fifo),
        .m_prev       (rf_m_prev),
        .l_prev       (l_prev_ld_reg),    // ★ Latched, not rf_l_prev direct ★
        .m_new_out    (m_new_out),
        .l_new_out    (l_new_out),
        .x_out        (x_out),
        .y_out        (y_out),
        .val_a_out    (val_a_out),
        .x_out_valid  (x_out_valid),
        .y_out_valid  (y_out_valid)
    );

    // ========================================================================
    // 8. O_prev Assembly Register (4096-bit / 8-chunk)
    // ========================================================================
    logic [4095:0] o_prev_assembly_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_prev_assembly_reg <= 4096'h0;
        end else if (o_prev_ld_en) begin
            o_prev_assembly_reg[o_prev_chunk_sel * 512 +: 512] <= o_prev_chunk_internal;
        end
    end

    // Unpack to FP32×128 for MAC SV init
    logic [31:0] o_prev_unpacked [0:127];
    generate
        for (genvar i = 0; i < 128; i++) begin : gen_o_prev
            assign o_prev_unpacked[i] = o_prev_assembly_reg[i*32 +: 32];
        end
    endgenerate

    // ========================================================================
    // 9. MAC SV Input MUX (★All FP32★)
    // ========================================================================
    always_comb begin
        if (acc_init) begin
            // Init cycle: X * O_prev
            sv_scalar_in = x_out;                        // FP32
            sv_init_mode = 1'b1;
            for (int k = 0; k < 128; k++)
                sv_vec_in[k] = o_prev_unpacked[k];       // FP32
        end else begin
            // Accumulation: Y_j * V_j
            sv_scalar_in = y_out;                        // FP32
            sv_init_mode = 1'b0;
            for (int k = 0; k < 128; k++)
                sv_vec_in[k] = v_vec_fp32[k];            // ★FP32 (casted)★
        end
    end

    // ---- 7.7 MAC Array SV ----
    mac_array_sv u_mac_array_sv (
        .clk           (clk),
        .rst_n         (rst_n),
        .scalar_in     (sv_scalar_in),
        .vec_in        (sv_vec_in),
        .valid_in      (sv_v_valid | acc_init),
        .init_mode     (sv_init_mode),
        .shadow_latch  (shadow_latch),
        .chunk_sel     (chunk_sel),
        .o_chunk_out   (o_chunk_out),
        .valid_out     (sv_valid_out)
    );

    // ========================================================================
    // 10. State RF (M/L/O Storage)
    // ========================================================================
    // rd_chunk_sel MUX: FSM o_prev_chunk_sel during loading, external otherwise
    logic [2:0] rf_rd_chunk_sel;

    always_comb begin
        if (o_prev_ld_en)
            rf_rd_chunk_sel = o_prev_chunk_sel;   // FSM-driven during O_prev load
        else
            rf_rd_chunk_sel = rd_chunk_sel;        // External diagnostic
    end

    state_rf u_state_rf (
        .clk              (clk),
        .rst_n            (rst_n),
        .row_cnt          (row_cnt),
        .o_wr_row         (o_wr_row),
        .exception_det_o  (1'b0),           // No exception handler in smoke top
        // M port
        .we_m             (we_m),
        .m_new_in         (m_new_out),
        // L port
        .we_l             (we_l),
        .l_new_in         (l_new_out),
        // O port
        .we_o             (we_o),
        .wr_chunk_sel     (chunk_sel),
        .o_chunk_in       (o_chunk_out),
        // Read ports
        .rd_chunk_sel     (rf_rd_chunk_sel),
        .m_prev_out       (rf_m_prev),
        .l_prev_out       (rf_l_prev),
        .o_prev_chunk_out (o_prev_chunk_internal)
    );

    // ========================================================================
    // 11. Diagnostic Output Wiring
    // ========================================================================
    assign m_prev_out       = rf_m_prev;
    assign l_prev_out       = rf_l_prev;
    assign o_chunk_read_out = o_prev_chunk_internal;

endmodule
