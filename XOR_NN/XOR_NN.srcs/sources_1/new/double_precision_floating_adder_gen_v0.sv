`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 26.04.2025 10:28:35
// Design Name: Floating Point Adder V0
// Module Name: double_precision_floating_adder_gen_v0
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: First Generation of Adder for Double Precision Floating Point Numbers
// 
// Dependencies: provide precision and mantisa number of bits
// 
// Revision: 0.02 Improved logic for calculating total shift, and adjusted bit field sizes. Corrected typo for ALU input.
// Revision 0.01 - File Created
// Additional Comments: In this version I am checking the correctness of barrel shifter and ALU for mantissa, LZD and exponent adder.
// 
//////////////////////////////////////////////////////////////////////////////////

module double_precision_floating_adder_gen_v0 (
    input [63:0] A,
    input [63:0] B,
    input [2:0] operation, // 1 for A + B, 2 for A - B, 0 for invalid 
    output logic [63:0] Res
);

logic [10:0] A_exponent, B_exponent;
logic [51:0] A_mantissa, B_mantissa;
logic A_sign, B_sign;

// Extracting the components of the double precision floating point numbers
assign A_exponent = A[62:52];
assign B_exponent = B[62:52];
assign A_mantissa = A[51:0];
assign B_mantissa = B[51:0];
assign A_sign = A[63];
assign B_sign = B[63];

logic [52:0] A_mantissa_pretended, B_mantissa_pretended;

assign A_mantissa_pretended = {1'b1, A_mantissa}; // Pretend the leading 1 for normalized numbers
assign B_mantissa_pretended = {1'b1, B_mantissa}; // Pretend the leading 1 for normalized numbers

//Configuring Exponent ALU to get the difference of exponent and select signal

logic select; // select = 1 if A is smaller than B
logic [10:0] Diff_exponent; // Difference of exponent needed for barrel shifter   

expo_alu_gen_v0 exponent_alu (
    .A(A_exponent),
    .B(B_exponent),
    .select(select),
    .Diff(Diff_exponent)
);

//Configuring Barrel Shifter to shift the mantissa of A or B based on the exponent difference
// Note this barrel shifter is for initial alignment.
logic [52:0] in_mantissa_pretended_shifted/*, in_mantissa_pretended_not_shifted*/, out_mantissa_pretended_shifted;

assign in_mantissa_pretended_shifted = !select ? B_mantissa_pretended : A_mantissa_pretended; // Select the mantissa to shift based on the exponent difference
//assign in_mantissa_pretended_not_shifted = select ? B_mantissa_pretended : A_mantissa_pretended; // Select the mantissa to shift based on the exponent difference

//Shift right
barrel_shifter_gen_v0 #(.n(53), .direction(1)) barrel_shifter_align (
    .A(in_mantissa_pretended_shifted),
    .shift(Diff_exponent[5:0]),
    .A_shift(out_mantissa_pretended_shifted)
);

//Configuring ALU to add/subtract the mantissa of A and B based on the operation
// Note this ALU is for mantissa addition/subtraction.

logic [52:0] A_mantissa_in_alu, B_mantissa_in_alu;
logic A_sign_in_alu, B_sign_in_alu;
logic [53:0] Res_mantissa;
logic Res_sign;

assign A_sign_in_alu = !select ? A_sign : B_sign; // Select the sign based on the exponent difference
assign B_sign_in_alu = select ? A_sign : B_sign; // Select the sign based on the exponent difference
assign A_mantissa_in_alu = select ? B_mantissa_pretended : A_mantissa_pretended; 
assign B_mantissa_in_alu = out_mantissa_pretended_shifted;

mantissa_alu_gen_vo mantissa_alu (
    .A_mantissa_pretended(A_mantissa_in_alu),
    .B_mantissa_pretended(B_mantissa_in_alu),
    .A_sign(A_sign),
    .B_sign(B_sign),
    .select(select),
    .operation(operation),
    .Res_sign(Res_sign),
    .Res_mantissa(Res_mantissa)
);

//Configuring Leading Zero Detector to get the leading zeros of the result mantissa
// If [53] is 1, then left shift by 1, else right shift by leading_zeros
// This is to normalize the result mantissa.

logic [7:0] leading_zeros; // Leading zeros of the result mantissa

leading_zero_detector_gen_v0 lzd (
  .mantissa(Res_mantissa),
  .shift(leading_zeros)
);

//Configuring Barrel Shifter to shift the mantissa of the result based on the leading zeros

logic [51:0] Res_temp;
barrel_shifter_gen_v0 #(.n(52), .shift_max(8), .direction(0)) barrel_shifter_normalize (
    .A(Res_mantissa[51:0]),
    .shift(leading_zeros),
    .A_shift(Res_temp)
);
logic [51:0] Res_mantissa_right_shifted;

assign Res_mantissa_right_shifted = Res_mantissa << 1;


logic [10:0] Res_expo_test;
assign Res[51:0] = Res_mantissa[53] ?  Res_mantissa_right_shifted : Res_temp; // Shift left if the result mantissa is negative
assign Res[62:52] = select ? (B_exponent - leading_zeros) : (A_exponent - leading_zeros); // Adjust the exponent based on the leading zeros
assign Res[63] = Res_sign; // Set the sign bit based on the result sign
assign Res_expo_test = select ? (B_exponent - leading_zeros + 1'b1) : (A_exponent - leading_zeros + 1'b1); // Adjust the exponent based on the leading zeros


endmodule