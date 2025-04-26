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
    input [10:0] A,
    input [10:0] B,
    output select,
    output logic [10:0] Diff
    );

//default B is smaller than A

assign select = B > A ? 1'b1 : 1'b0; // select = 1 if A is smaller than B

always_comb begin
    if (!select) begin
        Diff = A - B;
    end else begin
        Diff = B - A;
    end
end

endmodule
