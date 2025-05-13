`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 09.05.2025 21:04:25
// Design Name: Floating Point Multiplier V0
// Module Name: parameterized_floating_mul_gen_v0
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: I will attemopt to create a parameterized floating point multiplier, I will try to add support for
//for both single and double precision floating point numbers, by introducing paramerters.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
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


module parameterized_floating_mul_gen_v0 #(parameter int mantisa = `mantisa, exponent = `exponent, n = `n)(
    input [n-1:0] A,
    input [n-1:0] B,
    input [2:0] op,
    input [n-1:0] Product
    );

logic [mantisa-1:0] A_mantisa, B_mantisa;
logic [exponent-1:0] A_exponent, B_exponent;
logic A_sign, B_sign;

assign A_sign = A[n-1];
assign B_sign = B[n-1];
assign A_exponent = A[n-2:n-exponent-1];
assign B_exponent = B[n-2:n-exponent-1];
assign A_mantisa = {A[mantisa-1:0]};
assign B_mantisa = {B[mantisa-1:0]};

logic [mantisa:0] A_mantisa_prefix, B_mantisa_prefix;
assign A_mantisa_prefix = {1'b1, A_mantisa};
assign B_mantisa_prefix = {1'b1, B_mantisa};

logic [exponent-1:0] Arth_op;

expo_alu_gen_v1 expo_alu_gen_v1_inst (
    .A(A),
    .B(B),
    .select(select),
    .Arth_op(Arth_op)
);
integer product_in_size = mantisa + 1;
logic [2*product_in_size-1:0] Product_noround;


Carry_Save_Multiplier #(product_in_size) Carry_Save_Multiplier_inst (
    .A(A_mantisa_prefix),
    .B(B_mantisa_prefix),
    .Product(Product_noround)
);

logic shift_left;

assign shift_left = Product_noround[2*product_in_size-1];

logic [2*product_in_size-1:0] Product_shifted_no_prefix;
assign Product_shifted = shift_left ? Product_noround [2*product_in_size:1] : Product_noround[2*product_in_size-1:0];

logic [product_in_size-1:0] Product_round, Product_truncated;
loigc round_off = Product_noround[n-1];

assign Product_truncated = Product_shifted[product_in_size-1:n];

always_comb begin : Product_round_off
    if (round_off) begin
        Product_round = Product_shifted[product_in_size-1:n] + 1;
    end else begin
        Product_round = Product_shifted[product_in_size-1:n];
    end
end

//assign Product_round = Product_noround[2*n+1:1];

endmodule
