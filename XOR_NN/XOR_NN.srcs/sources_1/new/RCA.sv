`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 08.05.2025 20:10:02
// Design Name: Vector merge based on RCA
// Module Name: RCA
// Project Name: XOR_NN
// Target Devices: Pynq-Z2
// Tool Versions: 
// Description: Vector merge based on RCA for carry save multiplier
// 
// Dependencies: Will eliminate the need for a carry ripple
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module RCA #(parameter n = 4) (
    input [n-2:0] A,
    input [n-2:0] B,
    input Cin,
    output [n-1:0] Sum,
    output Cout
    );

logic [n-2:1] C;
//assign C[0] = Cin;

genvar i;

generate
    for (i = 0; i < n; i = i + 1) begin : rca_gen
        if (i == 0) begin
            HA HA_init(
                .A(A[i]),
                .B(B[i]),
                .Sum(Sum[i]),
                .Cout(C[i+1])
            );
        end else if (i == n-1) begin
            HA HA_final(
                .A(Cin),
                .B(C[i-1]),
                .Sum(Sum[i]),
                .Cout(Cout)
            );
        end else begin
            FA FA(
                .A(A[i]),
                .B(B[i]),
                .Cin(C[i]),
                .Sum(Sum[i]),
                .Cout(C[i+1])
            );
        end
    end
endgenerate



endmodule
