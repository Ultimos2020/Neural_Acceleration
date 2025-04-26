`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 26.04.2025 10:28:35
// Design Name: Floating Point Adder V0
// Module Name: barrel_shifter_gen_v0
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: First Generation of Barrel Shifter, without interleafing, pipelining and dual rails
// 
// Dependencies: provide precision and mantisa number of bits
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: I am not considering opto in this generation, I shall focus only on functionality. Internal parameters are not yet stated.
// 
//////////////////////////////////////////////////////////////////////////////////


module barrel_shifter_gen_v0 #(parameter int n = 64, int shift_max = $clog2(52), direction = 1) (
    input [n-1:0] A,
    input [shift_max-1:0] shift,
    //input direction, // 0 for left, 1 for right
    output logic [n-1:0] A_shift
    );

// Block to generate control signals for MUX stages
logic control_stage_0; //32
logic control_stage_1; //16
logic control_stage_2; //8
logic control_stage_3; //4
logic control_stage_4; //2
logic control_stage_5; //1

// Control signal generation
assign control_stage_0 = shift[5];
assign control_stage_1 = shift[4];
assign control_stage_2 = shift[3];
assign control_stage_3 = shift[2];
assign control_stage_4 = shift[1];
assign control_stage_5 = shift[0];

// Block to generate MUX inputs
logic [n-1:0] mux_input_0, mux_output_0;
logic [n-1:0] mux_input_1, mux_output_1;
logic [n-1:0] mux_input_2, mux_output_2;
logic [n-1:0] mux_input_3, mux_output_3;
logic [n-1:0] mux_input_4, mux_output_4;
logic [n-1:0] mux_input_5, mux_output_5;

// MUX stage 0
//shift by 32 bits
assign mux_input_0 = A;
assign mux_output_0 = control_stage_0 ? ( direction ?  mux_input_0 >> 32 : mux_input_0 << 32): mux_input_0;
assign mux_input_1 = mux_output_0;

// MUX stage 1
//shift by 16 bits
assign mux_output_1 = control_stage_1 ? ( direction ? mux_input_1 >> 16 :  mux_input_1 << 16): mux_input_1;
assign mux_input_2 = mux_output_1;

// MUX stage 2
//shift by 8 bits
assign mux_output_2 = control_stage_2 ? ( direction ? mux_input_2 >> 8 : mux_input_2 << 8): mux_input_2;
assign mux_input_3 = mux_output_2;

// MUX stage 3
//shift by 4 bits
assign mux_output_3 = control_stage_3 ? ( direction ? mux_input_3 >> 4 : mux_input_3 << 4): mux_input_3;
assign mux_input_4 = mux_output_3;

// MUX stage 4
//shift by 2 bits
assign mux_output_4 = control_stage_4 ? ( direction ? mux_input_4 >> 2 : mux_input_4 << 2): mux_input_4;
assign mux_input_5 = mux_output_4;

// MUX stage 5
// The last stage of the MUX, which shifts by 1 bit
assign mux_output_5 = control_stage_5 ? ( direction ? mux_input_5 >> 1 : mux_input_5 << 1): mux_input_5;

// Final output assignment
assign A_shift = mux_output_5;

    
endmodule

