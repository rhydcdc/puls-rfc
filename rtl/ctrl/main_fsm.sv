`timescale 1ns / 1ps
// =============================================================================
// Module : main_fsm
// Project: H9-O32 Flash-SFU
// Description:
//   Main FSM controller for the H9-O32 architecture.
//   Full redesign from legacy 324-cycle to 347-cycle timing (S3 full pipeline).
//
//   Key changes from legacy:
//     - global_cycle upper limit: 324 -> 332 -> 333 -> 346 (S3 full pipeline)
//     - QK latency: 4 -> 5 -> 9 cycles (S3: +5 from full adder tree split)
//     - EXP latency: 9 -> 13 -> 22 cycles (S3: +9 from Horner mult/add split)
//     - Total offset: +14 (QK +5, EXP +9) on acc_init, we_l, shadow_latch, we_o
//     - O Write-Back: 4-chunk (pc=15~18) -> 8-chunk (pc=20~27) -> 8-chunk (pc=33,0~6, WRAP!)
//     - O_prev Load:  4-chunk (pc=8~11)  -> 8-chunk (pc=10~17) -> 8-chunk (pc=24~31)
//     - chunk_sel / o_prev_chunk_sel: 2-bit -> 3-bit
//     - ★ o_wr_row latch: shadow_latch 시점에 write 대상 row 스냅샷 (we_o pc wrap 버그 수정)
//
//   gc bounds verified against C++ occupancy model (h9_o32_cycle_sim.cpp):
//     Row r base_gc = r * 34,  gc=0 corresponds to Timeline Cycle 1.
//     Row 0 (B=1): base_gc=0,  Row 7 (B=239): base_gc=238.
// =============================================================================

module main_fsm #(
    parameter MAX_TILES = 1
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        fifo_empty,     // Score FIFO empty 상태

    // ---- Host Interface ----
    input  logic        start,
    input  logic [7:0]  num_tiles,
    output logic        tile_ready,

    // ---- SRAM Control ----
    output logic        bank_sel,

    // ---- Sub-module Control Signals ----
    output logic        start_row,
    output logic        s_valid_in,
    output logic        fifo_rd_en,
    output logic        exp_x_en,
    output logic        exp_y_en,
    output logic        acc_init,
    output logic        we_m,
    output logic        we_l,
    output logic        shadow_latch,
    output logic        we_o,
    output logic        q_update,
    output logic        o_prev_ld_en,
    output logic [2:0]  o_prev_chunk_sel,   // 3-bit (8-chunk)
    output logic [2:0]  row_cnt_out,
    output logic [2:0]  chunk_sel,           // 3-bit (8-chunk)
    output logic [2:0]  o_wr_row             // ★ S3: O Write target row (latched at shadow_latch)
);

    // =========================================================================
    // State Machine Definitions
    // =========================================================================
    typedef enum logic [1:0] {
        ST_IDLE      = 2'd0,
        ST_RUN       = 2'd1,
        ST_TILE_DONE = 2'd2,
        ST_ALL_DONE  = 2'd3
    } state_t;

    state_t state, next_state;

    // =========================================================================
    // Internal Counters
    // =========================================================================
    logic [8:0] global_cycle;   // 0 ~ 346
    logic [5:0] phase_cnt;      // 0 ~ 33 (34-cycle period, architectural constant)
    logic [2:0] row_cnt;        // 0 ~ 7
    logic [7:0] tile_cnt;       // 0 ~ num_tiles-1

    // =========================================================================
    // Next State Logic (Combinational)
    // =========================================================================
    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start)
                    next_state = ST_RUN;
            end
            ST_RUN: begin
                if (global_cycle == 9'd346)
                    next_state = ST_TILE_DONE;
            end
            ST_TILE_DONE: begin
                if (tile_cnt == num_tiles - 8'd1)
                    next_state = ST_ALL_DONE;
                else
                    next_state = ST_RUN;
            end
            ST_ALL_DONE: begin
                if (start)
                    next_state = ST_RUN;
            end
            default: next_state = ST_IDLE;
        endcase
    end

    // =========================================================================
    // Sequential State and Counter Updates
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            global_cycle <= '0;
            phase_cnt    <= '0;
            row_cnt      <= '0;
            tile_cnt     <= '0;
            bank_sel     <= 1'b0;
        end else begin
            state <= next_state;

            case (state)
                ST_IDLE: begin
                    global_cycle <= '0;
                    phase_cnt    <= '0;
                    row_cnt      <= '0;
                    if (start)
                        tile_cnt <= '0;
                end

                ST_RUN: begin
                    if (global_cycle < 9'd346)
                        global_cycle <= global_cycle + 9'd1;

                    if (phase_cnt == 6'd33) begin
                        phase_cnt <= '0;
                        row_cnt   <= row_cnt + 3'd1;  // 3-bit natural wrap 0~7
                    end else begin
                        phase_cnt <= phase_cnt + 6'd1;
                    end
                end

                ST_TILE_DONE: begin
                    global_cycle <= '0;
                    phase_cnt    <= '0;
                    row_cnt      <= '0;
                    bank_sel     <= ~bank_sel;   // Ping-Pong toggle

                    if (tile_cnt < num_tiles - 8'd1)
                        tile_cnt <= tile_cnt + 8'd1;
                end

                ST_ALL_DONE: begin
                    if (start) begin
                        global_cycle <= '0;
                        phase_cnt    <= '0;
                        row_cnt      <= '0;
                        tile_cnt     <= '0;
                    end
                end
            endcase
        end
    end

    // =========================================================================
    // Port Assignments
    // =========================================================================
    assign row_cnt_out = row_cnt;
    assign tile_ready  = (state == ST_ALL_DONE) || (state == ST_TILE_DONE && tile_cnt == num_tiles - 8'd1);

    // =========================================================================
    // Combinational Control Signal Rules  [★ S3 Full Pipeline: +5 QK, +9 EXP = +14 total ★]
    // =========================================================================
    //
    // gc bound derivation (Row r base_gc = r*34):
    //   Signal fires at local cycle offset "ofs" => Row r fires at gc = r*34 + ofs.
    //   Row 0 gc = ofs,  Row 7 gc = 238 + ofs.
    //   gc_lower = ofs (Row 0 first fire),  gc_upper = 238 + ofs (Row 7 last fire).
    //   phase_cnt (pc) provides fine-grained filtering within each 34-cycle period.
    //
    //   S3 offsets vs E-folder baseline:
    //     Score-dependent signals: +5 (QK 4→9)
    //     EXP-dependent signals:  +14 (QK +5, Horner +9)
    //

    // ---- 0. q_update: Q vector latch (pc=0, perfectly aligned with K arrival) ----
    //   Local Cy2: gc_ofs=0, pc=0. Row 0: gc=0. Row 7: gc=238.  [UNCHANGED]
    assign q_update = (state == ST_RUN && phase_cnt == 6'd0 && global_cycle <= 9'd238);

    // ---- 1. start_row: Running-max reset trigger (pc=4) ----
    //   Local Cy5: gc_ofs=4, pc=4. Row 0: gc=4. Row 7: gc=242.  [UNCHANGED]
    assign start_row = (state == ST_RUN && phase_cnt == 6'd4 && global_cycle <= 9'd242);

    // ---- 2. s_valid_in: Score valid for running-max comparison ---- [S3: QK 9-cycle]
    //   K_0 Score: gc=1+9=10, pc=10.  K_31 Score: gc=32+9=41, pc=7.
    //   Active 32/34 cycles: pc=10..33, 0..7 (skip pc=8 we_m, pc=9).
    //   Row 0: gc=10..41. Row 7: gc=248..280.
    assign s_valid_in = (state == ST_RUN && global_cycle >= 9'd10 && global_cycle <= 9'd280 &&
                         (phase_cnt >= 6'd10 || phase_cnt <= 6'd7));

    // ---- 3. we_m + exp_x_en: M write-back & EXP_X trigger (pc=8) ---- [S3: +5]
    //   Local: gc_ofs=42, pc=8. Row 0: gc=42. Row 7: gc=280.
    assign we_m     = (state == ST_RUN && global_cycle >= 9'd42 && global_cycle <= 9'd280 &&
                       phase_cnt == 6'd8);
    assign exp_x_en = we_m;

    // ---- 4. fifo_rd_en: FIFO read for EXP_Y input ---- [S3: +5]
    //   32 cycles: pc=11,...,33,0,...,8 (skip pc=9,10).
    //   Row 0: gc=45..76. Row 7: gc=283..314.
    //   !fifo_empty guard로 Ghost Read 원천 차단.
    assign fifo_rd_en = (state == ST_RUN && global_cycle >= 9'd45 && global_cycle <= 9'd314 &&
                         (phase_cnt >= 6'd11 || phase_cnt <= 6'd8))
                        && !fifo_empty;

    // ---- 5. exp_y_en: EXP_Y pipeline valid ----
    //   FIFO Read와 동일 사이클에 EXP_Y valid 투입.
    //   FIFO→SUB_Y→EXP 경로가 전부 조합논리이므로 같은 사이클 정합이 정확.
    //   fifo_empty guard가 fifo_rd_en에서 상속되어 Ghost EXP_Y 원천 차단.
    assign exp_y_en = fifo_rd_en;

    // ---- 6. acc_init: L accumulator init (Y_0 arrival) ---- [S3: +14]
    //   Local: gc_ofs=66, pc=32. Row 0: gc=66. Row 7: gc=304.
    assign acc_init = (state == ST_RUN && global_cycle >= 9'd66 && global_cycle <= 9'd304 &&
                       phase_cnt == 6'd32);

    // ---- 7. we_l: L write-back ---- [S3: +14]
    //   Local: gc_ofs=99, pc=31. Row 0: gc=99. Row 7: gc=337.
    //   (L finalized at Cy99 end -> WB at Cy100)
    assign we_l = (state == ST_RUN && global_cycle >= 9'd99 && global_cycle <= 9'd337 &&
                   phase_cnt == 6'd31);

    // ---- 8. shadow_latch: O vector shadow register latch ---- [S3: +14]
    //   Local: gc_ofs=100, pc=32. Row 0: gc=100. Row 7: gc=338.
    //   (O finalized at Cy100 end -> Shadow at Cy101)
    assign shadow_latch = (state == ST_RUN && global_cycle >= 9'd100 && global_cycle <= 9'd338 &&
                           phase_cnt == 6'd32);

    // ---- 9. we_o: O vector 8-chunk serialized write-back ---- [S3: +14, pc WRAP!]
    //   Local: gc_ofs=101..108, pc=33,0,1,...,6. Row 0: gc=101..108. Row 7: gc=339..346.
    //   ★ pc wraps across 34-cycle boundary! pc=33 is chunk 0, pc=0~6 are chunks 1~7.
    //   ★ row_cnt changes at pc=33→0 boundary, so state_rf uses o_wr_row (latched) instead.
    assign we_o = (state == ST_RUN && global_cycle >= 9'd101 && global_cycle <= 9'd346 &&
                   (phase_cnt == 6'd33 || phase_cnt <= 6'd6));

    // chunk_sel: pc=33→idx0, pc=0→idx1, ..., pc=6→idx7
    logic [5:0] phase_for_chunk;
    assign phase_for_chunk = (phase_cnt == 6'd33) ? 6'd0 : (phase_cnt + 6'd1);
    assign chunk_sel = we_o ? phase_for_chunk[2:0] : 3'd0;

    // ---- 10. o_prev_ld_en: O_prev 8-chunk pre-load ---- [S3: +14, no wrap]
    //   Local: gc_ofs=58..65, pc=24..31. Row 0: gc=58..65. Row 7: gc=296..303.
    //   Note: Last chunk (pc=31) is exactly 1 cycle before acc_init (pc=32).
    assign o_prev_ld_en = (state == ST_RUN && global_cycle >= 9'd58 && global_cycle <= 9'd304 &&
                           phase_cnt >= 6'd24 && phase_cnt <= 6'd31);

    // o_prev_chunk_sel: 3-bit index derived from phase_cnt offset
    logic [5:0] phase_minus_24;
    assign phase_minus_24 = phase_cnt - 6'd24;
    assign o_prev_chunk_sel = phase_minus_24[2:0];

    // =========================================================================
    // ★ S3 NEW: O Write Row Address Latch (we_o pc wrap 버그 수정)
    // =========================================================================
    //   S3에서 we_o의 pc가 33→0→...→6으로 34-cycle 경계를 넘어 wrap됩니다.
    //   pc=33에서 row_cnt가 다음 행으로 증가하므로, pc=0~6에서의
    //   state_rf O Write 주소(lo_wr_addr = row_cnt - 2)가 잘못된 행을 가리킵니다.
    //   해결: shadow_latch 시점(pc=32, row_cnt 변경 전)에 올바른 write 주소를 스냅샷.
    //   하드웨어 비용: FF 3개.
    logic [2:0] o_wr_row_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            o_wr_row_reg <= 3'd0;
        else if (shadow_latch)
            o_wr_row_reg <= row_cnt - 3'd2;   // 이 시점에 row_cnt는 아직 안 바뀜
    end
    assign o_wr_row = o_wr_row_reg;

endmodule
