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
// Revision:0.02 Modification to approx output, and introduce capability for double and single precision
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


    //-----------------> i number of stages
    //|
    //|
    //|
    //|
    //|
    //|
    //|
    //|
    //V
    // j size of stage

`ifdef single_precision
    `define mantisa 23
    `define exponent 8
    `define n 32
`elsif double_precision
    `define mantisa 52
    `define exponent 11
    `define n 64
`else
    //for testing purposes
    `define mantisa 4
    `define exponent 0
    `define n 4
`endif

ifndef approx
    `define pro_size n
else
    `define pro_size 2*n
`endif

module Carry_Save_Multiplier #(parameter int n = `n)(
    input [n-1:0] A,
    input [n-1:0] B,
    output logic [pro_size-1:0] Product
    );

generate
    if (`aprrox) begin
                // This is the case for approximation
        logic [n-1:0] P [n-1:1]; // Partial products
        logic [n-1:0] S [n-1:0]; // Sum
        logic [n-1:0] C [n-1:1]; // Carry
        //ogic check;

        assign S[0][n-1:0] = A & {n{B[0]}};
        //assign C[0] = {(n){1'b0}};
        //assign P[0] = {n{1'b0}};
        genvar i, j;

        generate
            for (i = 1; i < n; i = i + 1) begin : pp_gen
                assign P[i] = A & {n{B[i]}};
            end

            for (i = 1; i < n; i = i + 1) begin : cs_gen
            if (i == 1) begin
                for (j = 0; j < n; j = j + 1) begin : ha_init
                if (j == n-1) begin
                            HA HA_init_last(
                                .A(P[i][j]),
                                .B(1'b0),
                                .Sum(S[i][j]),
                                .Cout(C[i][j])
                            );
                end else begin
                            HA HA_init(
                                .A(P[i][j]),
                                .B(S[i-1][j+1]),
                                .Sum(S[i][j]),
                                .Cout(C[i][j])
                            );
                    end 
                    end
            end else begin
                for (j = 0; j < n; j = j + 1) begin : fa_gen
                    if (j == n-1) begin
                        HA HA_last(
                            .A(P[i][j]),
                            .B(C[i-1][j]),
                            .Sum(S[i][j]),
                            .Cout(C[i][j])
                        );
                    end else begin
                        FA FA_bulk(
                            .A(P[i][j]),
                            .B(S[i-1][j+1]),
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
            .Sum(Product[2*n-1:n])
        );

        //assign Product[n-1:0] = S[n-1:0][1];
        genvar k;
        generate
            for (k = 0; k < n; k = k + 1) begin : product_assign
                assign Product[k] = S[k][0];
            end
        endgenerate
    
    end
    else begin
        // This is the case for no approximation
        logic [n-1:0] P [n-1:1]; // Partial products
        logic [n-1:0] S [n-1:0]; // Sum
        logic [n-1:0] C [n-1:1]; // Carry
        //ogic check;

        assign S[0][n-1:0] = A & {n{B[0]}};
        //assign C[0] = {(n){1'b0}};
        //assign P[0] = {n{1'b0}};
        genvar i, j;

        generate
            for (i = 1; i < n; i = i + 1) begin : pp_gen
                assign P[i] = A & {n{B[i]}};
            end

            for (i = 1; i < n; i = i + 1) begin : cs_gen
            if (i == 1) begin
                for (j = 0; j < n; j = j + 1) begin : ha_init
                if (j == n-1) begin
                            HA HA_init_last(
                                .A(P[i][j]),
                                .B(1'b0),
                                .Sum(S[i][j]),
                                .Cout(C[i][j])
                            );
                end else begin
                            HA HA_init(
                                .A(P[i][j]),
                                .B(S[i-1][j+1]),
                                .Sum(S[i][j]),
                                .Cout(C[i][j])
                            );
                    end 
                    end
            end else begin
                for (j = 0; j < n; j = j + 1) begin : fa_gen
                    if (j == n-1) begin
                        HA HA_last(
                            .A(P[i][j]),
                            .B(C[i-1][j]),
                            .Sum(S[i][j]),
                            .Cout(C[i][j])
                        );
                    end else begin
                        FA FA_bulk(
                            .A(P[i][j]),
                            .B(S[i-1][j+1]),
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
            .Sum(Product[2*n-1:n])
        );

        //assign Product[n-1:0] = S[n-1:0][1];
        genvar k;
        generate
            for (k = 0; k < n; k = k + 1) begin : product_assign
                assign Product[k] = S[k][0];
            end
        endgenerate

    end
endgenerate


endmodule
