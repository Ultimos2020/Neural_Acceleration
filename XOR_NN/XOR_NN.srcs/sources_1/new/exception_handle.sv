`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 06.05.2025 08:22:56
// Design Name: First Generation of excecption handling
// Module Name: exception_handle
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: First Generation of excecption handling for inputs and outputs
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module exception_handle(
    input [63:0] A,
    input [63:0] B,
    input [63:0] Res,
    output [2:0] inf,
    output [2:0] zero,
    output [2:0] denorm,
    output [2:0] NaN,
    output [2:0] exception_detect
    );

logic mantissa_zero_A, exponent_zero_A, exponent_one_A;
logic mantissa_zero_B, exponent_zero_B, exponent_one_B;
logic mantissa_zero_Res, exponent_zero_Res, exponent_one_Res;

assign mantissa_zero_A = ~|A[51:0];
assign exponent_zero_A = ~|A[62:52];
assign exponent_one_A = &A[62:52];
assign mantissa_zero_B = ~|B[51:0];
assign exponent_zero_B = ~|B[62:52];
assign exponent_one_B = &B[62:52];
assign mantissa_zero_Res = ~|Res[51:0];
assign exponent_zero_Res = ~|Res[62:52];
assign exponent_one_Res = &Res[62:52];


assign inf[0] = (exponent_one_A && mantissa_zero_A) ? 1'b1 : 1'b0;
assign inf[1] = (exponent_one_B && mantissa_zero_B) ? 1'b1 : 1'b0;
assign inf[2] = (exponent_one_Res && mantissa_zero_Res) ? 1'b1 : 1'b0;

assign zero[0] = (exponent_zero_A && mantissa_zero_A) ? 1'b1 : 1'b0;
assign zero[1] = (exponent_zero_B && mantissa_zero_B) ? 1'b1 : 1'b0;
assign zero[2] = (exponent_zero_Res && mantissa_zero_Res) ? 1'b1 : 1'b0;

assign denorm[0] = (exponent_zero_A && ~mantissa_zero_A) ? 1'b1 : 1'b0;
assign denorm[1] = (exponent_zero_B && ~mantissa_zero_B) ? 1'b1 : 1'b0;
assign denorm[2] = (exponent_zero_Res && ~mantissa_zero_Res) ? 1'b1 : 1'b0;

assign NaN[0] = (exponent_one_A && ~mantissa_zero_A) ? 1'b1 : 1'b0;
assign NaN[1] = (exponent_one_B && ~mantissa_zero_B) ? 1'b1 : 1'b0;
assign NaN[2] = (exponent_one_Res && ~mantissa_zero_Res) ? 1'b1 : 1'b0;


assign exception_detect = {|{inf[2],zero[2], denorm[2], NaN[2]}, 
                            |{inf[1],zero[1], denorm[1], NaN[1]},
                            |{inf[0],zero[0], denorm[0], NaN[0]}};
endmodule
