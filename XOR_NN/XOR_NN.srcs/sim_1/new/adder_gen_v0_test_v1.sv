`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05.05.2025 21:56:40
// Design Name: 
// Module Name: add_gen_v0_test_v1
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


module add_gen_v0_test_v1();


  //----------------------------------------------------------------------  
  // Signals
  //----------------------------------------------------------------------  
  logic [63:0] A, B;
  logic [2:0]  operation;   // 1 = add, 2 = sub, others = invalid
  logic [63:0] Res;
  integer      errors;
  bit         mismatch;

  //----------------------------------------------------------------------  
  // DUT instantiation
  //----------------------------------------------------------------------  
  double_precision_floating_adder_gen_v0 adder_DUT (
    .A         (A),
    .B         (B),
    .operation (operation),
    .Res       (Res)
  );

  //----------------------------------------------------------------------  
  // Task to drive inputs, wait, and check against expected
  //----------------------------------------------------------------------  
  task run_case(
    input [63:0] a,
    input [63:0] b,
    input [2:0]  op,
    input [63:0] expected,
    input string name
  );
    begin
      mismatch = 0;
      A         = a;
      B         = b;
      operation = op;
      #2; // wait for result

      if (Res != expected) begin
        errors = errors + 1;
        mismatch = 1;
        //#10;
        $display(" [%0s] ERROR: A=0x%016h, B=0x%016h, op=%0d → got 0x%016h, expected 0x%016h",
                 name, a, b, op, Res, expected);
        
      end else begin
        errors = errors + 0;
        mismatch = 0;
        //#10;
        $display(" [%0s] PASS: A=0x%016h, B=0x%016h, op=%0d → got 0x%016h, expected 0x%016h",
                  name, a, b, op, Res, expected);
        
      end
    end
    #10;
  endtask

  //----------------------------------------------------------------------  
  // Test sequence (normal finite cases only)
  //----------------------------------------------------------------------  
  initial begin
    errors = 0;

  

    // 5.0 – 2.0 = 3.0
    run_case(64'h4014_0000_0000_0000,
             64'h4000_0000_0000_0000,
             3'b010,
             64'h4008_0000_0000_0000,
             "5-2");

    // +500 - 100 = +400
    run_case(64'h407F400000000000,
             64'h4059000000000000,
             3'b010,
             64'h4079000000000000,
             "500-100");

    // +500 - 1000 = -500
    run_case(64'h407F400000000000,
             64'h408F400000000000,
             3'b010,
             64'hC07F400000000000,
             "500-1000");

    // +500 + 1000 = +1500
    run_case(64'h407F400000000000,
             64'h408F400000000000,
             3'b001,
             64'h4097700000000000,
             "500+1000");

    //152 - 100 = 52
    run_case(64'h4063000000000000,
             64'h4059000000000000,
             3'b010,
             64'h404A000000000000,
             "152-100");

    //36 - 100 = -64

    run_case(64'h4042000000000000,
             64'h4059000000000000,
             3'b010,
             64'hC050000000000000,
             "36-100");

    //-99 + 100 = 1
    run_case(64'hC058C00000000000,
             64'h4059000000000000,
             3'b001,
             64'h3FF0000000000000,
             "-99+100");

    //-99 - 100 = -199
    run_case(64'hC058C00000000000,
             64'h4059000000000000,
             3'b010,
             64'hC068E00000000000,
             "-99-100"); 

     // 2.0 + 10.0 = 12.0
    run_case(64'h4000_0000_0000_0000,
             64'h4024_0000_0000_0000,
             3'b001,
             64'h4028_0000_0000_0000,
             "2+10");                                   

    // Summary
    if (errors) begin
      $display("\n*** TEST FAILED: %0d ERROR(S) ***", errors);
      $fatal;
    end else begin
      $display("\n*** ALL TESTS PASSED ***");
      $finish;
    end
  end


endmodule
