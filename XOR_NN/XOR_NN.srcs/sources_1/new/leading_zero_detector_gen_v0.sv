`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 26.04.2025 19:16:53
// Design Name: Leading Zero Detector (LZD)
// Module Name: leading_zero_detector_gen_v0
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: Priority Encoder based leading zero detection
// 
// Dependencies: 
// 
// Revision: 0.03 - Retracted substraction change.
// Revision 0.02 - Improved logic for calculating total shift.
// Revision 0.01 - File Created
// Additional Comments: Current implementation is yet to be tested for maximum shift magitude.
// 
//////////////////////////////////////////////////////////////////////////////////


module leading_zero_detector_gen_v0(
    input [53:0] mantissa,
    output [7:0] shift
    );

logic group_0, group_1, group_2, group_3, group_4, group_5, group_6;
logic [7:0] group_sel;
logic [2:0] group_val;
logic [2:0] group_intra_val;
logic [6:0] leading_1;

assign group_0 = | mantissa[7:0];
assign group_1 = | mantissa[15:8];
assign group_2 = | mantissa[23:16];
assign group_3 = | mantissa[31:24];
assign group_4 = | mantissa[39:32];
assign group_5 = | mantissa[47:40];
assign group_6 = | mantissa[53:48];
// group_7 means all zeros

always_comb begin : Inter_group_sel
    if (group_6) begin
        group_sel = {2'b00,mantissa[53:48]};
        group_val = 3'b110;
    end else if (group_5) begin
        group_sel = mantissa[47:40];
        group_val = 3'b101;
    end else if (group_4) begin
        group_sel = mantissa[39:32];
        group_val = 3'b100;
    end else if (group_3) begin
        group_sel = mantissa[31:24];
        group_val = 3'b011;
    end else if (group_2) begin
        group_sel = mantissa[23:16];
        group_val = 3'b010;
    end else if (group_1) begin
        group_sel = mantissa[15:8];
        group_val = 3'b001;
    end else if (group_0) begin
        group_sel = mantissa[7:0];
        group_val = 3'b000;
    end else begin
        group_sel = 7'b0000000; // all zeros
        group_val = 3'b111; // all zeros
    end
    
end

always_comb begin : Intra_group_sel
    if (group_sel[7]) begin
        group_intra_val = 3'b111;
    end else if (group_sel[6]) begin
        group_intra_val = 3'b110;
    end else if (group_sel[5]) begin
        group_intra_val = 3'b101;
    end else if (group_sel[4]) begin
        group_intra_val = 3'b100;
    end else if (group_sel[3]) begin
        group_intra_val = 3'b011;
    end else if (group_sel[2]) begin
        group_intra_val = 3'b010;
    end else if (group_sel[1]) begin
        group_intra_val = 3'b001;
    end else if (group_sel[0]) begin
        group_intra_val = 3'b000;
    end else begin
        group_intra_val = 3'b000; // all zeros
    end
end

assign leading_1 = {4'b0000, group_intra_val} + {1'b0,group_val, 3'b000};
assign shift = 7'd52 - leading_1;

endmodule
