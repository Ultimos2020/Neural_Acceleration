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
integer errors;
integer add_errors, mul_errors;
logic mismatch;
logic [7:0] expected;

Carry_Save_Multiplier #(4) CS_Mul (
    .A(A),
    .B(B),
    .Product(Res)
);

initial begin
    errors = 0;
    add_errors = 0;
    mul_errors = 0;
    mismatch = 0;
    for (i = 10; i < 16; i = i + 1) begin
        for (j = 0; j < 10; j = j + 1) begin
            A = i;
            B = j;
            expected = A * B;
            #5;
            if (Res !== expected) begin
                errors++;
                mismatch = 1;
                $display("Mismatch: A: %b, B: %b, Expected: %b, Got: %b", A, B, expected, Res);
                if (mismatch) begin
                    if (Res[3:0] !== expected[3:0]) begin
                        $display("Mul failure, expected: %b, got: %b", expected[3:0], Res[3:0]);
                        mul_errors++;
                    end
                    if (Res[7:4] !== expected[7:4]) begin
                        $display("Add failure, expected: %b, got: %b", expected[7:4], Res[7:4]);
                        if (Res[7] !== expected[7]) begin
                            $display("Add failure 7, expected: %b, got: %b", expected[7], Res[7]);
                        end
                        if (Res[6] !== expected[6]) begin
                            $display("Add failure 6, expected: %b, got: %b", expected[6], Res[6]);
                        end
                        if (Res[5] !== expected[5]) begin
                            $display("Add failure 5, expected: %b, got: %b", expected[5], Res[5]);
                        end
                        if (Res[4] !== expected[4]) begin
                            $display("Add failure 4, expected: %b, got: %b", expected[4], Res[4]);
                        end
                        add_errors++;
                    end
                end
            end else begin
                mismatch = 0;
            end
            //$display("A: %b, B: %b, Product: %b", A, B, Res);
            

        end
    end
    if (errors > 0) begin
        $display("Test failed with %d errors.", errors);
        $display("Mul errors: %d, Add errors: %d", mul_errors, add_errors);
    end else begin
        $display("All tests passed.");
        
    end
    $finish;
end



endmodule
