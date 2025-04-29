`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 27.04.2025 17:50:22
// Design Name: Testbench for Floating Point Adder V0 
// Module Name: adder_gen_v0_test
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: First Generation of Testbench for Floating Point Adder V0. just to check coorectness of the design.
// with focus on the ALU for mantissa, LZD and exponent adder and barrel shifter.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module adder_gen_v0_test();

logic [63:0] A, B;
logic [2:0] operation; // 1 for A + B, 2 for A - B, 0 for invalid
logic [63:0] Res;

 double_precision_floating_adder_gen_v0 adder_DUT (
    .A(A),
    .B(B),
    .operation(operation),
    .Res(Res)
);

initial begin
    // Test case 1: A + B
    A = 64'h4000000000000000; // 2.0
    B = 64'h4024000000000000; // 10.0
    operation = 3'b001; // Addition
    #10;
    
    // Test case 2: A - B
    B = 64'h4000000000000000; // 2.0
    A = 64'h4014000000000000; // 5.0
    operation = 3'b010; // Subtraction
    #10;

    // Test case 3: Invalid operation
    A = 64'h4000000000000000; // 2.0
    B = 64'h4024000000000000; // 10.0
    operation = 3'b000; // Invalid operation
    #10;

    $stop;
end


endmodule
