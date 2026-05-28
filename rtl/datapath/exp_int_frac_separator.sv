`timescale 1ns / 1ps

// =============================================================================
// Module: exp_int_frac_separator
// Project: Flash-SFU H9-O32
// Description: Stage 2 전반. y = x * log2(e) 결과를 정수부 n(5-bit)과 소수부 f(Q0.10)로 분리.
// Pipeline Latency: Combinational
// =============================================================================

module exp_int_frac_separator (
    input  logic [15:0] x_fp16,      // y 값 (x * log2 e의 결과, FP16)
    output logic [4:0]  int_part,    // 분리된 정수부 n (최종 지수 스케일링용, 5-bit 2의 보수)
    output logic [9:0]  f_fixed      // 분리된 소수부 f (Q0.10 고정소수점, 항상 ≥ 0, exp_fp16_normalizer로 전달)
);

    // ----------------------------------------------------------------------
    // 1. 입력 FP16의 각 필드 분리
    // ----------------------------------------------------------------------
    logic       sign;
    logic [4:0] exp;
    logic [9:0] frac_bits;
    
    assign sign      = x_fp16[15];
    assign exp       = x_fp16[14:10];
    assign frac_bits = x_fp16[9:0];

    // ----------------------------------------------------------------------
    // 2-1. 소수점 위치 결정을 위한 디코더 로직 (비트 마스킹)
    // ----------------------------------------------------------------------
    logic [9:0] sel;
    
    assign sel[9] = ~exp[4];
    assign sel[8] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2] & ~exp[1] & ~exp[0]);
    assign sel[7] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2] & ~exp[1]);
    assign sel[6] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2] & ~exp[1])
                            | (exp[4] & ~exp[3] & ~exp[2] &  exp[1] & ~exp[0]);
    assign sel[5] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2] & ~exp[1])
                            | (exp[4] & ~exp[3] & ~exp[2] &  exp[1]);
    assign sel[4] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2])
                            | (exp[4] & ~exp[3] &  exp[2] & ~exp[1] & ~exp[0]);
    assign sel[3] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2])
                            | (exp[4] & ~exp[3] &  exp[2] & ~exp[1]);
    assign sel[2] = ~exp[4] | (exp[4] & ~exp[3] & ~exp[2])
                            | (exp[4] & ~exp[3] &  exp[2] & ~exp[1])
                            | (exp[4] & ~exp[3] &  exp[2] &  exp[1] & ~exp[0]);
    assign sel[1] = ~exp[4] | ~exp[3];
    assign sel[0] = ~exp[4] | (exp[4] & ~exp[3]) 
                            | (exp[4] &  exp[3] & ~exp[2] & ~exp[1] & ~exp[0]);

    // ----------------------------------------------------------------------
    // 2-2. MUX x 10: 소수부 비트 추출 (나머지는 오른쪽 0 패딩)
    // ----------------------------------------------------------------------
    logic [9:0] frac_extracted;
    
    assign frac_extracted[9] = sel[9] ? frac_bits[9] : 1'b0;
    assign frac_extracted[8] = sel[8] ? frac_bits[8] : 1'b0;
    assign frac_extracted[7] = sel[7] ? frac_bits[7] : 1'b0;
    assign frac_extracted[6] = sel[6] ? frac_bits[6] : 1'b0;
    assign frac_extracted[5] = sel[5] ? frac_bits[5] : 1'b0;
    assign frac_extracted[4] = sel[4] ? frac_bits[4] : 1'b0;
    assign frac_extracted[3] = sel[3] ? frac_bits[3] : 1'b0;
    assign frac_extracted[2] = sel[2] ? frac_bits[2] : 1'b0;
    assign frac_extracted[1] = sel[1] ? frac_bits[1] : 1'b0;
    assign frac_extracted[0] = sel[0] ? frac_bits[0] : 1'b0;

    // ----------------------------------------------------------------------
    // 2-3. 소수부 정렬 (frac_aligned 계산)
    // ----------------------------------------------------------------------
    logic [10:0] full_mantissa;
    assign full_mantissa = {1'b1, frac_bits};

    logic [9:0] frac_aligned;
    always_comb begin
        if (exp < 5'd15)
            frac_aligned = full_mantissa >> (5'd15 - exp);  // hidden-1 포함 우측 시프트
        else
            frac_aligned = frac_extracted << (exp - 5'd15); // 소수부 비트 좌측 정렬
    end

    // ----------------------------------------------------------------------
    // 2-4. floor 조건 판단 (frac_exists, floor_adjust)
    // ----------------------------------------------------------------------
    // E < 15: 값 전체가 소수부이므로 hidden-1 포함 full_mantissa가 존재하면 소수 있음
    // E >= 15: 추출된 소수부 비트로 판단
    // BUG FIX: 오른쪽 시프트(>>)로 인해 잘려나가 0이 된 미세한 소수부(Underflow)는
    // 수학적으로는 존재하더라도 레지스터 상 0이 되므로, 논리적 동기화를 위해 frac_exists=0 으로 판정하여
    // n = -1, f = 0.0 으로 이격되는 50% 반토막 버그를 방어합니다.
    logic frac_exists;
    assign frac_exists = (frac_aligned != 10'd0);

    logic floor_adjust;
    assign floor_adjust = sign & frac_exists; // 음수이면서 소수부가 있으면 floor 추가 조정 (-1)

    // ----------------------------------------------------------------------
    // 2-5. 정수부 n 및 소수부 f최종 확정 (with 5-Bit Saturation)
    // ----------------------------------------------------------------------
    // 정수부 추출을 위한 중간 신호 (오버플로우 방지를 위해 6-bit 할당)
    logic [5:0] int_bits_by_exp;
    always_comb begin
        if (exp < 5'd15) begin
            int_bits_by_exp = 6'd0;
        end else begin
            case (exp)
                5'd15: int_bits_by_exp = 6'd1;
                5'd16: int_bits_by_exp = 6'd2 + {5'd0, frac_bits[9]};
                5'd17: int_bits_by_exp = 6'd4 + {4'd0, frac_bits[9:8]};
                5'd18: int_bits_by_exp = 6'd8 + {3'd0, frac_bits[9:7]};
                5'd19: int_bits_by_exp = 6'd16 + {2'd0, frac_bits[9:6]};
                default: int_bits_by_exp = 6'd31; // exp >= 20일 때 최대 31로 클리핑
            endcase
        end
    end

    // 부호 부여 및 Floor 처리를 위해 7-bit signed 사용
    // [V2 UPGRADE] 명시적인 2의 보수 변환으로 삼항 연산 시의 부호 꼬임(Wrap-around) 원천 차단
    logic [6:0] n_mag;
    assign n_mag = {1'b0, int_bits_by_exp}; // 7-bit 양수 절댓값
    
    logic signed [6:0] n_sgn;
    assign n_sgn = sign ? $signed(~n_mag + 7'd1) : $signed(n_mag);

    logic signed [6:0] n_raw;
    assign n_raw = floor_adjust ? (n_sgn - 7'sd1) : n_sgn; // floor 조정 (뺄셈기)

    // 5-bit [-16, +15] 범위로 완전 새츄레이션 (비상 차단기)
    logic signed [4:0] n_final;
    always_comb begin
        // [V2 UPGRADE] 양단에 $signed() 명시적 캐스팅으로 Unsigned 강제 형변환 언더플로우 버그 완벽 방어!
        if ($signed(n_raw) > $signed(7'sd15))
            n_final = 5'sd15;      // 거대 양수는 +15로 고정 (Inf 유도)
        else if ($signed(n_raw) < $signed(-7'sd16))
            n_final = -5'sd16;     // 거대 음수(-17 이하)는 무조건 -16으로 꽉 묶음 (Flush-to-Zero 유도)
        else
            n_final = n_raw[4:0];  // 정상 범위는 그대로 패스
    end

    assign int_part = n_final;

    logic [9:0] frac_inverted;
    assign frac_inverted = ~frac_aligned;              // 비트 반전 (NOT 게이트 x 10)

    logic [9:0] frac_two_comp;
    assign frac_two_comp = frac_inverted + 10'd1;      // 1-frac 연산 완료 (10-bit RCA)

    // floor_adjust 조건(음수+소수)이면 2의 보수 결과를, 양수거나 정수면 정렬된 소수부 사용
    assign f_fixed = floor_adjust ? frac_two_comp : frac_aligned;

endmodule
