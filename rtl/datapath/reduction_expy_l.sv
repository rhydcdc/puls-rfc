`timescale 1ns / 1ps

// =============================================================================
// Module: reduction_expy_l
// Project: Flash-SFU H9-O32
// Description: Reduction Unit lower half — val_A(X×l_prev) latch, EXP_Y 32-way
//              pipeline (S_j − m_new → exp → FP32), and L accumulator.
//              Key change from legacy: x_out/y_out are FP32, no FP16→FP32 casters.
// Pipeline Latency: 13 cycles (EXP_Y path, from exp_y_en to y_out_valid)
// =============================================================================

module reduction_expy_l (
    input  logic        clk,
    input  logic        rst_n,

    // ---- FSM Control Signals ----
    input  logic        exp_y_en,      // EXP_Y 투입 유효 (Cy40~71, 32사이클)
    input  logic        acc_init,      // L 누적기 초기화 (Cy53, 1사이클)

    // ---- Data Inputs (from 07 모듈) ----
    input  logic [31:0] x_out,         // ★FP32★ X 스케일링 팩터
    input  logic        x_out_valid,   // X 산출 valid

    // ---- Data Inputs (from State RF / FIFO) ----
    input  logic [15:0] s_out_q,       // FIFO에서 인출된 S_j
    input  logic [15:0] m_y_reg,       // 안정된 m_new (from 07 모듈)
    input  logic [31:0] l_prev,        // 이전 타일의 L (FP32, from State RF)

    // ---- Data Outputs ----
    output logic [31:0] y_out,         // ★FP32★ Y_j (EXP 결과)
    output logic        y_out_valid,   // Y_j valid
    output logic [31:0] l_new_out,     // L 최종 누적값 (FP32)
    output logic [31:0] val_a_out      // val_A_reg 값 (디버깅/MAC SV용)
);

    // ═══════════════════════════════════════════════════════════════════
    // 1. val_A 계산 & 래치
    //    X × l_prev = FP32 × FP32 → FP32
    //    ★ x_out이 이미 FP32이므로 fp16_to_fp32_caster 불필요 ★
    // ═══════════════════════════════════════════════════════════════════
    logic [31:0] mult_xl_out;
    logic [31:0] val_A_reg;

    fp32_mult u_mult_xl (
        .a      (x_out),       // FP32 직접 연결
        .b      (l_prev),      // FP32
        .result (mult_xl_out)
    );

    // [FF 1: val_A_reg] X가 나온 사이클(Cy51)에 래치
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)           val_A_reg <= 32'h0;
        else if (x_out_valid) val_A_reg <= mult_xl_out;
    end

    assign val_a_out = val_A_reg;

    // ═══════════════════════════════════════════════════════════════════
    // 2. EXP_Y 경로: (S_j − m_new) → EXP 13사이클 → FP32 Y_j
    //    ★ y_out이 이미 FP32이므로 fp16_to_fp32_caster 불필요 ★
    // ═══════════════════════════════════════════════════════════════════

    // 뺄셈: S_j - m_new (FP16 부호 반전 + 덧셈)
    logic [15:0] neg_m_y_reg;
    assign neg_m_y_reg = {~m_y_reg[15], m_y_reg[14:0]};

    logic [15:0] sub_y_out;
    fp16_add u_sub_y (
        .a      (s_out_q),
        .b      (neg_m_y_reg),
        .result (sub_y_out)
    );

    // EXP_Y 13사이클 파이프라인 → FP32 출력
    exp_top u_exp_y (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (exp_y_en),
        .x_in      (sub_y_out),
        .valid_out (y_out_valid),
        .exp_out   (y_out)         // ★FP32 [31:0]★
    );

    // ═══════════════════════════════════════════════════════════════════
    // 3. L 누적기 (val_A + Σ Y_j)
    //    acc_init_delay: acc_init이 y_out_valid보다 1사이클 빌리 도착
    //    → 1사이클 늦춰야 y_out_valid와 동기화됨
    // ═══════════════════════════════════════════════════════════════════
    logic [31:0] l_acc_reg;
    logic [31:0] add_l_in_a;
    logic [31:0] add_l_out;

    // acc_init_delay: 1사이클 지연으로 y_out_valid와 정렬
    logic acc_init_delay;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_init_delay <= 1'b0;
        else        acc_init_delay <= acc_init;
    end

    // MUX: acc_init_delay이면 val_A_reg (첫 누적), 아니면 이전 누적값
    assign add_l_in_a = acc_init_delay ? val_A_reg : l_acc_reg;

    // FP32 덧셈기: L += Y_j
    // ★ y_out이 이미 FP32이므로 캐스터 없이 직접 연결 ★
    fp32_add u_add_l (
        .a      (add_l_in_a),
        .b      (y_out),       // ★FP32 직접 연결★
        .result (add_l_out)
    );

    // L 누적 레지스터: y_out_valid일 때만 갱신
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)           l_acc_reg <= 32'h0;
        else if (y_out_valid) l_acc_reg <= add_l_out;
    end

    assign l_new_out = l_acc_reg;

    // ═══════════════════════════════════════════════════════════════════
    // Inter Trace Points (synthesis off)
    // ═══════════════════════════════════════════════════════════════════
    // synthesis translate_off
    always @(posedge clk) begin
        if (x_out_valid) begin
            $display("[VAL_A_CALC] time=%0t | x_out=%h l_prev=%h mult_xl_out=%h",
                     $time, x_out, l_prev, mult_xl_out);
            $display("[VAL_A_REG]  time=%0t | val_A_reg=%h",
                     $time, mult_xl_out);
        end
        if (exp_y_en) begin
            $display("[SUB_Y]     time=%0t | s_out_q=%h m_y_reg=%h sub_y_out=%h",
                     $time, s_out_q, m_y_reg, sub_y_out);
        end
        if (y_out_valid) begin
            $display("[EXP_Y_OUT] time=%0t | y_out=%h (FP32)",
                     $time, y_out);
            $display("[L_ACC]     time=%0t | l_acc_reg=%h add_l_in_a=%h y_out=%h add_l_out=%h",
                     $time, l_acc_reg, add_l_in_a, y_out, add_l_out);
        end
        if (acc_init_delay) begin
            $display("[L_ACC_INIT] time=%0t | val_A_reg=%h y_out=%h add_l_out=%h",
                     $time, val_A_reg, y_out, add_l_out);
        end
    end
    // synthesis translate_on

endmodule
