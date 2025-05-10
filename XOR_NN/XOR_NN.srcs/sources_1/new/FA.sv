`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 08.05.2025 19:00:48
// Design Name: Full Adder
// Module Name: FA
// Project Name: XOR_NN
// Target Devices: Pynq-Z2
// Tool Versions: 
// Description: Basic full adder module
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module FA(
    input A,
    input B,
    input Cin,
    output Sum,
    output Cout
    );

logic Sum1, Cout1, Cout2;

HA HA1(
    .A(A),
    .B(B),
    .Sum(Sum1),
    .Cout(Cout1)
);
HA HA2(
    .A(Sum1),
    .B(Cin),
    .Sum(Sum),
    .Cout(Cout2)
);

assign Cout = Cout1 | Cout2;

endmodule
