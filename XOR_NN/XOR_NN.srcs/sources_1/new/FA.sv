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
`ifdef approx
    `define ports , output logic Cout
    `define gen_approx 1
`else
    `define ports , output logic Sum, Cout
    `define gen_approx 0
`endif

module FA #(parameter approx = `gen_approx) (
    input A,
    input B,
    input Cin
    `ports
    );

logic Cout1, Cout2;

generate
    if (approx == 1) begin
        
        HA HA1_approx(
        .A(A),
        .B(B),
        .Cout(Cout1)
        );
        HA HA2_approx(
        .A(Sum1),
        .B(Cin),
        .Cout(Cout2)
        );

    end else begin
        logic Sum1;
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
    end
endgenerate

assign Cout = Cout1 | Cout2;

endmodule
