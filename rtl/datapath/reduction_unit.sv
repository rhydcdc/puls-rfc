`timescale 1ns / 1ps

// =============================================================================
// Module: reduction_unit
// Project: Flash-SFU H9-O32
// Description: Top-level Reduction Unit wrapping 07(reduction_m_expx) and
//              08(reduction_expy_l). Pure wiring — no internal logic.
//              EXP instances: 2 (EXP_X inside 07, EXP_Y inside 08).
//              Key change from legacy: x_out/y_out are FP32 (not FP16).
// Pipeline Latency: 13 cycles (EXP path, from exp_x/y_en to x/y_out_valid)
// =============================================================================

module reduction_unit (
    input  logic        clk,
    input  logic        rst_n,

    // ---- FSM Control ----
    input  logic        start_row,     // Row 시작 (l_prev snapshot 등 — 07 내부에서는 미사용)
    input  logic        we_m,          // M write-back (pc=8) → 07 내부에서 1cyc 지연 후 running_max 리셋
    input  logic        s_valid_in,    // Score 입력 유효
    input  logic        exp_x_en,      // EXP_X 시작 (Cy38)
    input  logic        exp_y_en,      // EXP_Y 투입 (Cy40~71)
    input  logic        acc_init,      // L 누적 초기화 (Cy53)

    // ---- Data Inputs ----
    input  logic [15:0] s_in,          // 실시간 S_j (from MAC QK)
    input  logic [15:0] s_out_q,       // 지연된 S_j (from Score FIFO)
    input  logic [15:0] m_prev,        // 이전 max (from State RF)
    input  logic [31:0] l_prev,        // 이전 L (FP32, from l_prev_ld_reg)

    // ---- Data Outputs ----
    output logic [15:0] m_new_out,     // 확정된 m_new → State RF
    output logic [31:0] l_new_out,     // L 최종값 → State RF (FP32)
    output logic [31:0] x_out,         // ★FP32★ X → MAC SV
    output logic [31:0] y_out,         // ★FP32★ Y_j → MAC SV
    output logic [31:0] val_a_out,     // val_A → 디버깅용

    // ---- Valid Outputs ----
    output logic        x_out_valid,
    output logic        y_out_valid
);

    // ═══════════════════════════════════════════════════════════════════
    // Internal Wires: 07 → 08 interconnect
    // ═══════════════════════════════════════════════════════════════════
    logic [15:0] m_y_reg_int;  // m_new 래치 (EXP_Y 뺄셈용)

    // ═══════════════════════════════════════════════════════════════════
    // 07: m_new Finder + EXP_X
    // ═══════════════════════════════════════════════════════════════════
    reduction_m_expx u_m_expx (
        .clk          (clk),
        .rst_n        (rst_n),
        .start_row    (start_row),
        .we_m         (we_m),          // ★ Phase5-3: pc=9 reset trigger source ★
        .s_valid_in   (s_valid_in),
        .exp_x_en     (exp_x_en),
        .s_in         (s_in),
        .m_prev       (m_prev),
        .m_new_out    (m_new_out),
        .x_out        (x_out),
        .m_y_reg_out  (m_y_reg_int),   // ★ 내부 연결 → 08 입력 ★
        .x_out_valid  (x_out_valid)
    );

    // ═══════════════════════════════════════════════════════════════════
    // 08: val_A + EXP_Y + L 누적기
    // ═══════════════════════════════════════════════════════════════════
    reduction_expy_l u_expy_l (
        .clk          (clk),
        .rst_n        (rst_n),
        .exp_y_en     (exp_y_en),
        .acc_init     (acc_init),
        .x_out        (x_out),         // ★ 07 출력 직결 ★
        .x_out_valid  (x_out_valid),   // ★ 07 출력 직결 ★
        .s_out_q      (s_out_q),
        .m_y_reg      (m_y_reg_int),   // ★ 내부 연결 from 07 ★
        .l_prev       (l_prev),
        .y_out        (y_out),
        .y_out_valid  (y_out_valid),
        .l_new_out    (l_new_out),
        .val_a_out    (val_a_out)
    );

endmodule
