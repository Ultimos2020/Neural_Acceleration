`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NA
// Engineer: Ratnesh Mohan
// 
// Create Date: 26.04.2025 21:42:49
// Design Name: ALU for Mantissa of Floating Point Numbers
// Module Name: mantissa_alu_gen_vo
// Project Name: XOR_NN
// Target Devices: Pynq Z2
// Tool Versions: 
// Description: First Generation of ALU for Mantissa of Floating Point Numbers
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: If this version I have basically configurted all scenarios of addition and subtraction. 
//I have not yet considered the case of overflow, and other exceptions. In future I hope to add those as well.
//Along with better handling of the sign bit & operation mode.
// 
//////////////////////////////////////////////////////////////////////////////////


module mantissa_alu_gen_vo(
    input [52:0] A_mantissa_pretended,
    input [52:0] B_mantissa_pretended,
    input A_sign,
    input B_sign,
    input select, //0 for A > B , 1 for B > A
    input [2:0] operation, // 1 for A + B, 2 for A - B, 0 for invalid
    output logic Res_sign,
    output logic [53:0] Res_mantissa
    );

logic [52:0] A_mantissa_complemented;
logic [52:0] B_mantissa_complemented;
logic [53:0] Res_mantissa_temp;

assign A_mantissa_complemented = ~A_mantissa_pretended + 1'b1; // 2's complement
assign B_mantissa_complemented = ~B_mantissa_pretended + 1'b1; // 2's complement
always_comb begin
    if (operation == 3'b001) begin // A + B
        case ({A_sign, B_sign})
        2'b00: begin // A + B
            Res_mantissa_temp = A_mantissa_pretended + B_mantissa_pretended;
            Res_mantissa = Res_mantissa_temp;
            Res_sign = 1'b0;
        end
        2'b01: begin // A - B
            Res_mantissa_temp = A_mantissa_pretended + B_mantissa_complemented;
            Res_mantissa = select ? ~Res_mantissa_temp + 1'b1 : Res_mantissa_temp;
            Res_sign = select ? 1'b1 : 1'b0;
        end
        2'b10: begin // B - A
            Res_mantissa_temp = B_mantissa_pretended + A_mantissa_complemented;
            Res_mantissa = ~select ? ~Res_mantissa_temp + 1'b1 : Res_mantissa_temp;
            Res_sign = ~select ? 1'b1 : 1'b0;
        end
        2'b11: begin // -A - B
            Res_mantissa_temp = A_mantissa_complemented + B_mantissa_complemented;
            Res_mantissa = ~Res_mantissa_temp + 1'b1;
            Res_sign = 1'b1;
        end
        default: begin
            Res_mantissa_temp = 54'b0; // Invalid operation
            Res_mantissa = 54'b0; // Invalid operation
            Res_sign = 1'b0;
        end
        endcase
    end else if (operation == 3'b010) begin // A - B
        case ({A_sign, B_sign})
        2'b00: begin // A - B
            Res_mantissa_temp = A_mantissa_pretended + B_mantissa_complemented;
            Res_mantissa = select ? ~Res_mantissa_temp + 1'b1 : Res_mantissa_temp;
            Res_sign = select ? 1'b1 : 1'b0;
        end
        2'b01: begin // A + B
            Res_mantissa_temp = B_mantissa_pretended + A_mantissa_pretended;
            Res_mantissa = Res_mantissa_temp;
            Res_sign = 1'b0;
        end
        2'b10: begin // -A - B
            Res_mantissa_temp = A_mantissa_complemented + B_mantissa_complemented;
            Res_mantissa = ~Res_mantissa_temp + 1'b1;
            Res_sign = 1'b1;
        end
        2'b11: begin // -A + B
            Res_mantissa_temp = A_mantissa_complemented + B_mantissa_pretended;
            Res_mantissa = ~select ? ~Res_mantissa_temp + 1'b1 : Res_mantissa_temp;
            Res_sign = ~select ? 1'b1 : 1'b0;
        end
        default: begin
            Res_mantissa = 54'b0; // Invalid operation
            Res_sign = 1'b0; // Invalid operation
            Res_mantissa_temp = 54'b0; // Invalid operation
        end
        endcase
    end else begin
        Res_mantissa = 54'b0; // Invalid operation
        Res_sign = 1'b0; // Invalid operation
        Res_mantissa_temp = 54'b0; // Invalid operation
    end
end



endmodule
