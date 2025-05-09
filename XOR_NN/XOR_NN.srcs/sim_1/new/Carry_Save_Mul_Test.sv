`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.05.2025 20:49:24
// Design Name: 
// Module Name: Carry_Save_Mul_Test
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Carry_Save_Mul_Test(
);

logic [3:0] A, B;
logic [7:0] Res;
integer i, j;
logic mismatch;
logic [7:0] expected;

Carry_Save_Multiplier #(4) CS_Mul (
    .A(A),
    .B(B),
    .Product(Res)
);

initial begin
    for (i = 0; i < 16; i = i + 1) begin
        for (j = 0; j < 16; j = j + 1) begin
            A = i;
            B = j;
            expected = A * B;
            #5;
            if (Res !== expected) begin
                mismatch = 1;
                $display("Mismatch: A: %b, B: %b, Expected: %b, Got: %b", A, B, expected, Res);
                if (mismatch) begin
                    if (Res[3:0] !== expected[3:0]) begin
                        $display("Mul failure, expected: %b, got: %b", expected[3:0], Res[3:0]);
                    end
                    if (Res[7:4] !== expected[7:4]) begin
                        $display("Add failure, expected: %b, got: %b", expected[7:4], Res[7:4]);
                    end
                end
            end else begin
                mismatch = 0;
            end
            $display("A: %b, B: %b, Product: %b", A, B, Res);
            #15;

        end
    end
    $finish;
end

endmodule
