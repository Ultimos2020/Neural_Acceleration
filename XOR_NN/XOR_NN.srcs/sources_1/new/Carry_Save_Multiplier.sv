`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 08.05.2025 19:00:48
// Design Name: Carry Save Multiplier
// Module Name: Carry_Save_Multiplier
// Project Name: XOR_NN
// Target Devices: Pynq-Z2
// Tool Versions: 
// Description: First interation of carry save multiplier
// 
// Dependencies: Currently will use RCA which will be replaceed by a needed architecture.
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Carry_Save_Multiplier #(parameter n = 4)(
    input [n-1:0] A,
    input [n-1:0] B,
    output logic [2*n-1:0] Product
    );

logic [n-1:0] P [n-1:1]; // Partial products
logic [n-1:0] S [n-1:0]; // Sum
logic [n-1:0] C [n-1:1]; // Carry
logic check;

assign S[0] = A & {n{B[0]}};
//assign C[0] = {(n){1'b0}};
//assign P[0] = {n{1'b0}};
genvar i, j;

generate
    for (i = 1; i < n; i = i + 1) begin : pp_gen
        assign P[i] = A & {n{B[i]}};
    end

    for (i = 1; i < n; i = i + 1) begin : cs_gen
    if (i == 1) begin
        for (j = 0; j < n; j = j + 1) begin : fa_gen
                    HA HA_init(
                        .A(P[i][j]),
                        .B(S[i-1][j]),
                        .Sum(S[i][j]),
                        .Cout(C[i][j])
                    );
            end
    end else begin
        for (j = 0; j < n; j = j + 1) begin : fa_gen
            if (j == n-1) begin
                HA HA1(
                    .A(P[i][j]),
                    .B(C[i-1][j]),
                    .Sum(S[i][j]),
                    .Cout(C[i][j])
                );
            end else begin
                FA FA1(
                    .A(P[i][j]),
                    .B(S[i-1][j]),
                    .Cin(C[i-1][j]),
                    .Sum(S[i][j]),
                    .Cout(C[i][j])
                );
            end
        end
    end
end
endgenerate

RCA #(n) RCA_final(
    .A(S[n-1][n-1:1]),
    .B(C[n-1][n-2:0]),
    .Cin(C[n-1][n-1]),
    .Sum(Product[2*n-1:n]),
    .Cout(check)
);

//assign Product[n-1:0] = S[n-1:0][1];

always_comb begin : product_assign
    for (int i = 0; i < n; i = i + 1) begin
        Product[i] = S[i][0];
    end
end

endmodule
