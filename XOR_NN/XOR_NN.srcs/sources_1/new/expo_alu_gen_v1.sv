`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 12.05.2025 20:09:44
// Design Name: exponent ALU Generation v1
// Module Name: expo_alu_gen_v1
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: Its time to create a parameterized ALU for exponent of floating point numbers.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: Super Saiyan 3 is here.
// Testing for changes from iPAD
//////////////////////////////////////////////////////////////////////////////////
`ifdef single_precision
    `define mantisa 23
    `define exponent 8
    `define n 32
`elsif double_precision
    `define mantisa 52
    `define exponent 11
    `define n 64
`else
    //single_precision is default
    `define mantisa 23
    `define exponent 8
    `define n 32
`endif

`ifdef multiplier
    `define Arth 1'b0 
`elsif adder
    `define Arth 1'b1
`else
    //default is adder
    `define Arth 1'b1
`endif

module expo_alu_gen_v1 #(parameter int mantisa = `mantisa, exponent = `exponent, n = `n, Arth = `Arth)(
    input [n-1:0] A,
    input [n-1:0] B,
    output logic select,
    output logic [exponent-1:0] Arth_op
    );

logic equal;
logic [exponent-1:0] A_exponent, B_exponent;
assign A_exponent = A[n-2:n-exponent-1];
assign B_exponent = B[n-2:n-exponent-1];
assign equal = (A_exponent == B_exponent) ? 1'b1 : 1'b0; // equal = 1 if A is equal to B

always_comb begin
    if (equal) begin
        select = (A[mantisa-1:0] > B[mantisa-1:0]) ? 1'b0 : 1'b1; // if equal, select is 0
    end else begin
        select = (B_exponent > A_exponent) ? 1'b1 : 1'b0; // select = 0 if A is greater than B
    end
end

generate
    if (`Arth == 1'b0) begin
        logic carry; //to highlight overflow
        always_comb {carry, Arth_op} = A_exponent + B_exponent;
    end else begin
        always_comb begin
            if (!select) begin
                assign Arth_op = A_exponent - B_exponent;
            end else begin
                assign Arth_op = B_exponent - A_exponent;
            end
        end
    end
endgenerate


endmodule
