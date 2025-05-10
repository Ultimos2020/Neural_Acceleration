`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 08.05.2025 19:00:48
// Design Name: Half Adder
// Module Name: HA
// Project Name: XOR_NN
// Target Devices: Pynq-Z2
// Tool Versions: 
// Description: Basic half adder module
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module HA(
    input A,
    input B,
    output Sum,
    output Cout
    );

assign {Cout, Sum} = A + B;

endmodule
