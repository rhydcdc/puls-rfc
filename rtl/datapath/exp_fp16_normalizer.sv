`timescale 1ns / 1ps

// =============================================================================
// Module: exp_fp16_normalizer
// Project: Flash-SFU H9-O32
// Description: Stage 2 후반. 고정소수점 f (Q0.10) 값을 FP16 형태로 정규화.
// Pipeline Latency: Combinational
// =============================================================================

module exp_fp16_normalizer (
    input  logic [9:0]  f_fixed,    // 고정소수점 소수부 (Q0.10, 범위: 0 ~ 0.9990, 항상 ≥ 0)
    output logic [15:0] frac_fp16   // 정규화된 FP16 출력 (부호=0 보장, f=0이면 16'h0000)
);

    // -------------------------------------------------------------------------
    // ① 직렬 체인 Priority Encoder (C.3 Serial Chain 방식, 27 게이트)
    //    Pre_OR[k]: 비트 9~k+1 중 하나라도 1이 있으면 1
    //    y_k[k]   : 비트 k가 첫 번째 1이면 1 (One-hot)
    // -------------------------------------------------------------------------
    logic [9:0] pre_or;
    assign pre_or[9] = f_fixed[9];
    assign pre_or[8] = pre_or[9] | f_fixed[8];
    assign pre_or[7] = pre_or[8] | f_fixed[7];
    assign pre_or[6] = pre_or[7] | f_fixed[6];
    assign pre_or[5] = pre_or[6] | f_fixed[5];
    assign pre_or[4] = pre_or[5] | f_fixed[4];
    assign pre_or[3] = pre_or[4] | f_fixed[3];
    assign pre_or[2] = pre_or[3] | f_fixed[2];
    assign pre_or[1] = pre_or[2] | f_fixed[1];
    assign pre_or[0] = pre_or[1] | f_fixed[0];

    logic [9:0] y_k;
    assign y_k[9] = f_fixed[9];
    assign y_k[8] = f_fixed[8] & ~pre_or[9];
    assign y_k[7] = f_fixed[7] & ~pre_or[8];
    assign y_k[6] = f_fixed[6] & ~pre_or[7];
    assign y_k[5] = f_fixed[5] & ~pre_or[6];
    assign y_k[4] = f_fixed[4] & ~pre_or[5];
    assign y_k[3] = f_fixed[3] & ~pre_or[4];
    assign y_k[2] = f_fixed[2] & ~pre_or[3];
    assign y_k[1] = f_fixed[1] & ~pre_or[2];
    assign y_k[0] = f_fixed[0] & ~pre_or[1];

    // ② shift_k 인코딩 (y_k One-hot → 0~9 바이너리)
    //    shift_k = 9 - (첫 번째 1의 bit 인덱스)
    //    bit9가 1이면 shift_k=0, bit0만 1이면 shift_k=9
    logic [3:0] shift_k;
    always_comb begin
        case (1'b1)
            y_k[9] : shift_k = 4'd0;
            y_k[8] : shift_k = 4'd1;
            y_k[7] : shift_k = 4'd2;
            y_k[6] : shift_k = 4'd3;
            y_k[5] : shift_k = 4'd4;
            y_k[4] : shift_k = 4'd5;
            y_k[3] : shift_k = 4'd6;
            y_k[2] : shift_k = 4'd7;
            y_k[1] : shift_k = 4'd8;
            y_k[0] : shift_k = 4'd9;
            default: shift_k = 4'd0; // f_fixed == 0 (아래 예외 처리로 catch)
        endcase
    end

    // -------------------------------------------------------------------------
    // ② 10-bit Barrel Shifter (Left-Shift)
    //    shift_k번 좌측 시프트 → bit9(MSB)에 leading 1 안착
    // -------------------------------------------------------------------------
    logic [9:0] f_shifted;
    assign f_shifted = f_fixed << shift_k;
    // f_shifted[9]는 leading 1 (FP16에서 암묵적 생략)
    // f_shifted[8:0]이 9비트 가수부

    // -------------------------------------------------------------------------
    // ③ FP16 지수부 계산
    //    bit9가 첫 1 (shift_k=0) → f 값이 [0.5, 1.0) → 실제지수=-1 → E_f=14
    //    일반식: E_f = 14 - shift_k
    // -------------------------------------------------------------------------
    logic [4:0] f_exp;
    assign f_exp = 5'd14 - {1'b0, shift_k};
    // shift_k 최대 9 → E_f 최소 5 (정상 범위, 언더플로우 없음)
    // f_fixed가 0이 아닌 한 E_f >= 5로 보장

    // -------------------------------------------------------------------------
    // ④ FP16 재포장 + ⑤ 예외 처리 (f_fixed == 0)
    // -------------------------------------------------------------------------
    always_comb begin
        if (f_fixed == 10'd0) begin
            // f = 0.0: 입력이 완벽한 정수이거나 0 → FP16 0 (0x0000) 출력
            // exp_final_scaler에서 2^n × 0.0 = 0.0 처리됨
            frac_fp16 = 16'h0000;
        end else begin
            // 부호(0) | 지수(E_f) | 가수부(f_shifted[8:0] + 0패딩 1비트)
            // f_shifted[9] = leading 1 → 암묵적 생략
            // f_shifted[8:0] = 9비트 가수부 정수 비트들
            // 하위 1비트는 0으로 패딩 (Q0.10 → 10비트 가수부 중 9비트 사용)
            frac_fp16 = {1'b0, f_exp, f_shifted[8:0], 1'b0};
        end
    end

endmodule
