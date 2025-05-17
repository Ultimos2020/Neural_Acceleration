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
`ifdef approx
    `define ports , output logic Cout
    `define gen_approx 1
`else
    `define ports , output logic Sum, Cout
    `define gen_approx 0
`endif 

module HA #(parameter approx = `gen_approx) (
    input A,
    input B
    `ports
    );

generate
    if (approx == 1) begin
        assign Cout = A & B;
    end else begin
        assign {Cout, Sum} = A + B;
    end
endgenerate


endmodule
