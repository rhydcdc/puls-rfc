`timescale 1ns / 1ps

// =============================================================================
// Module: reduction_m_expx
// Project: Flash-SFU H9-O32
// Description: Reduction Unit upper half — Running Max finder, m_new decision,
//              (m_prev − m_new) subtraction, and EXP_X pipeline producing
//              FP32 scaling factor X.
//              Key change from legacy: x_out is FP32 (not FP16).
// Pipeline Latency: 13 cycles (EXP_X path, from exp_x_en to x_out_valid)
// =============================================================================

module reduction_m_expx (
    input  logic        clk,
    input  logic        rst_n,

    // ---- FSM Control Signals ----
    input  logic        start_row,     // (unused internally: reset trigger moved to we_m_d1)
    input  logic        we_m,          // M write-back (pc=8) → 1-cycle delayed → running_max reset (pc=9)
    input  logic        s_valid_in,    // Score 입력 유효
    input  logic        exp_x_en,      // EXP_X 시작 트리거 (Cy38)

    // ---- Data Inputs ----
    input  logic [15:0] s_in,          // 실시간 S_j (from mac_array_qk)
    input  logic [15:0] m_prev,        // 이전 타일의 max (from State RF)

    // ---- Data Outputs ----
    output logic [15:0] m_new_out,     // 확정된 m_new → State RF + EXP_Y 뺄셈
    output logic [31:0] x_out,         // ★FP32★ X 스케일링 팩터
    output logic [15:0] m_y_reg_out,   // EXP_Y 뺄셈용 안정된 m_new → 08 모듈로 전달

    // ---- Valid Outputs ----
    output logic        x_out_valid
);

    // ═══════════════════════════════════════════════════════════════════
    // 1. Running Max (m_new Finder)
    // ═══════════════════════════════════════════════════════════════════
    logic [15:0] running_max;
    logic [15:0] m_y_reg;
    logic        s_in_is_larger;

    // we_m 1-cycle delay → reset trigger at pc=9 (S3 QK 9-cycle safe window)
    logic        we_m_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) we_m_d1 <= 1'b0;
        else        we_m_d1 <= we_m;
    end

    // FP16 Fast Comparator (Combinational)
    // 부호 비교 → 크기 비교 (레거시 그대로)
    always_comb begin
        if (s_in[15] != running_max[15])
            s_in_is_larger = (s_in[15] == 1'b0);  // 양수가 더 큼
        else if (s_in[15] == 1'b0)
            s_in_is_larger = (s_in[14:0] > running_max[14:0]);
        else
            s_in_is_larger = (s_in[14:0] < running_max[14:0]);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running_max <= 16'hFC00;  // -Infinity
        end else begin
            if (we_m_d1)
                running_max <= m_prev;  // 새 Row 시작: m_prev로 리셋 (pc=9, K_31 반영 후)
            else if (s_valid_in && s_in_is_larger)
                running_max <= s_in;
        end
    end

    // ═══════════════════════════════════════════════════════════════════
    // 2. Running Max Forwarding (★Run7 Fix★)
    //    s_valid_in && exp_x_en 동시일 때, running_max 레지스터가
    //    아직 갱신 전이므로 조합논리로 실시간 비교 결과를 우회.
    // ═══════════════════════════════════════════════════════════════════
    wire [15:0] running_max_fwd;
    assign running_max_fwd = (s_valid_in && s_in_is_larger) ? s_in : running_max;

    // m_final_reg: 이전 row running_max 스냅샷 (we_m_d1 시점, K_31 반영 직후)
    logic [15:0] m_final_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            m_final_reg <= 16'hFC00;
        else if (we_m_d1)
            m_final_reg <= running_max;
    end

    // m_new 출력: exp_x_en 활성 시 실시간 포워딩, 아닐 때는 래치된 값
    assign m_new_out = exp_x_en ? running_max_fwd : m_final_reg;

    // ═══════════════════════════════════════════════════════════════════
    // 3. m_y_reg 래치
    //    EXP_Y가 EXP_X보다 2사이클 늦게 시작하므로,
    //    m_new를 안정적으로 유지하는 래치.
    // ═══════════════════════════════════════════════════════════════════
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            m_y_reg <= 16'hFC00;
        else if (exp_x_en)
            m_y_reg <= running_max_fwd;
    end

    assign m_y_reg_out = m_y_reg;

    // ═══════════════════════════════════════════════════════════════════
    // 4. EXP_X 경로: (m_prev − m_new) → EXP 13사이클 → FP32 x_out
    // ═══════════════════════════════════════════════════════════════════

    // 뺄셈: m_prev - m_new (FP16 부호 반전 + 덧셈)
    logic [15:0] neg_m_new;
    assign neg_m_new = {~m_new_out[15], m_new_out[14:0]};

    logic [15:0] sub_x_out;
    fp16_add u_sub_x (
        .a      (m_prev),
        .b      (neg_m_new),
        .result (sub_x_out)
    );

    // EXP 13사이클 파이프라인 → FP32 출력
    exp_top u_exp_x (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (exp_x_en),
        .x_in      (sub_x_out),
        .valid_out (x_out_valid),
        .exp_out   (x_out)         // ★FP32 [31:0]★
    );

    // ═══════════════════════════════════════════════════════════════════
    // Inter Trace Points (synthesis off)
    // ═══════════════════════════════════════════════════════════════════
    // synthesis translate_off
    always @(posedge clk) begin
        if (s_valid_in) begin
            $display("[RUNNING_MAX] time=%0t | s_in=%h running_max=%h s_in_is_larger=%b",
                     $time, s_in, running_max, s_in_is_larger);
        end
        if (exp_x_en) begin
            $display("[M_NEW] time=%0t | m_new_out=%h m_prev=%h running_max_fwd=%h",
                     $time, m_new_out, m_prev, running_max_fwd);
            $display("[SUB_X] time=%0t | sub_x_out=%h (m_prev=%h - m_new=%h)",
                     $time, sub_x_out, m_prev, m_new_out);
        end
        if (x_out_valid) begin
            $display("[EXP_X_OUT] time=%0t | x_out=%h (FP32)",
                     $time, x_out);
        end
    end
    // synthesis translate_on

endmodule
