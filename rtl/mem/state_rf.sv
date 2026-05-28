`timescale 1ns / 1ps

// =============================================================================
// Module: state_rf
// Project: Flash-SFU H9-O32
// Description: State Register File (M, L, O vectors), 8-chunk FP32 O_reg support
// Pipeline Latency: Combinational read, 1 cycle write
// =============================================================================

module state_rf (
    input  logic             clk,
    input  logic             rst_n,

    // FSM Target Row
    input  logic [2:0]       row_cnt,
    input  logic [2:0]       o_wr_row,         // ★ S3: FSM-latched O write target row (shadow_latch @ pc=32)

    // Exception
    input  logic             exception_det_o,

    // M Write (Cy38)
    input  logic             we_m,
    input  logic [15:0]      m_new_in,

    // L Write (Cy85)
    input  logic             we_l,
    input  logic [31:0]      l_new_in,

    // O Write (Cy87~94, 8-chunk)
    input  logic             we_o,
    input  logic [2:0]       wr_chunk_sel,     // 3bit 0~7
    input  logic [511:0]     o_chunk_in,       // FP32x16 = 512bit

    // Read Ports
    input  logic [2:0]       rd_chunk_sel,     // 3bit 0~7
    output logic [15:0]      m_prev_out,
    output logic [31:0]      l_prev_out,
    output logic [511:0]     o_prev_chunk_out  // FP32x16 = 512bit
);

    // =========================================================================
    // 1. Storage
    // =========================================================================
    logic [15:0]   M_reg [0:7];    // FP16
    logic [31:0]   L_reg [0:7];    // FP32
    logic [4095:0] O_reg [0:7];    // FP32x128 = 4096bit

    // =========================================================================
    // 2. Address Calculation
    // =========================================================================
    logic [2:0] rd_addr;
    logic [2:0] m_wr_addr;
    logic [2:0] lo_wr_addr;
    logic [2:0] m_rd_addr;

    always_comb begin
        rd_addr    = row_cnt - 3'd1;
        m_wr_addr  = row_cnt - 3'd1;
        lo_wr_addr = row_cnt - 3'd2;
        m_rd_addr  = we_m ? (row_cnt - 3'd1) : row_cnt;
    end

    // =========================================================================
    // 3. Exception Safe Values
    // =========================================================================
    localparam logic [15:0]  SAFE_M       = 16'hFC00;
    localparam logic [31:0]  SAFE_L       = 32'h00000000;
    localparam logic [511:0] SAFE_O_CHUNK = 512'h0;

    // =========================================================================
    // 4. Write-Back Logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 8; i++) begin
                M_reg[i] <= 16'hFC00;
                L_reg[i] <= 32'h0;
                O_reg[i] <= 4096'h0;
            end
        end else begin
            // M Write-Back
            if (we_m) begin
                M_reg[m_wr_addr] <= exception_det_o ? SAFE_M : m_new_in;
            end

            // L Write-Back
            if (we_l) begin
                L_reg[lo_wr_addr] <= exception_det_o ? SAFE_L : l_new_in;
            end

            // O Write-Back (8-chunk)
            // ★ S3: use o_wr_row (latched) instead of lo_wr_addr because we_o spans
            //   the pc=33→0 row_cnt wrap boundary (chunk 0 at pc=33, chunks 1~7 at pc=0~6).
            if (we_o) begin
                O_reg[o_wr_row][wr_chunk_sel * 512 +: 512] <=
                    exception_det_o ? SAFE_O_CHUNK : o_chunk_in;
            end
        end
    end

    // =========================================================================
    // 5. Read Logic
    // =========================================================================
    always_comb begin
        m_prev_out       = M_reg[m_rd_addr];
        l_prev_out       = L_reg[m_rd_addr];
        o_prev_chunk_out = O_reg[rd_addr][rd_chunk_sel * 512 +: 512];
    end

endmodule
