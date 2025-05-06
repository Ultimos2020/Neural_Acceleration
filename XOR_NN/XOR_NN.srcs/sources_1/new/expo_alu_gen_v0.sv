`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 26.04.2025 21:27:31
// Design Name: Difference Generator for Exponent of Floating Point Numbers 
// Module Name: expo_alu_gen_v0
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: Basic ALU for Exponent of Floating Point Numbers to geenrate diff and select signal.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module expo_alu_gen_v0(
    input [63:0] A,
    input [63:0] B,
    output logic select,
    //output equal,
    output logic [10:0] Diff
    );

//default B is smaller than A
logic equal;

//assign select = B[62:52] > A[62:52] ? 1'b1 : 1'b0; // select = 1 if A is smaller than B
assign equal = (A[62:52] == B[62:52]) ? 1'b1 : 1'b0; // equal = 1 if A is equal to B

always_comb begin
    if (equal) begin
        select = (A[51:0] > B[51:0]) ? 1'b0 : 1'b1;; // if equal, select is 0
    end else begin
        select = (B[62:52] > A[62:52]) ? 1'b1 : 1'b0; // select = 0 if A is greater than B
    end
end
 

always_comb begin
    if (!select) begin
        Diff = A[62:52] - B[62:52];
    end else begin
        Diff = B[62:52] - A[62:52];
    end
end

endmodule
