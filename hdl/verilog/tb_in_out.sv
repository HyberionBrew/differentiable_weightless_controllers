`timescale 1ns/1ps
`include "core_pkg.svh"

module tb_in_out;
  import core_pkg::*;

  // ------------------------------------------------------------------//
  //  Params / clock
  // ------------------------------------------------------------------//
  localparam int PIPE_LAT = 20; // enough margin for all settings, so nobody gets a heart attack

  logic clk = 0;
  always #5 clk = ~clk; // 100 MHz

  // ------------------------------------------------------------------//
  //  Signals to / from DUT
  // ------------------------------------------------------------------//
  logic [0:core_pkg::ENCODE_OBS_BIT_WIDTH-1] obs [0:core_pkg::OBS-1];   // inputs
  logic [0:ACTION_BITWIDTH-1]                act [0:ACTIONS-1];         // outputs
  logic [0:core_pkg::OBS*core_pkg::BITS_PER_OBS-1] thermo_bus;
  // ------------------------------------------------------------------//
  //  Device Under Test
  // ------------------------------------------------------------------//

  thermo_encode #(
    .OBS       (core_pkg::OBS),
    .BITS_PER_OBS(core_pkg::BITS_PER_OBS)
    ) u_enc (
        .quant_in   (obs),
        .thermo_out (thermo_bus)
  );
    
  dwn_top dut (
    .clk       (clk),
    .rstn      (1'b1),
    .in_bus    (thermo_bus),
    .group_sum (act)
  );


// ---------- DEBUG: add from here ----------
  localparam int SHOW_OBS  = core_pkg::OBS;
  localparam int SHOW_LUTS = core_pkg::LUTS;

  //task automatic print_thermo_block(input int idx);
  //  $display("  thermo[%0d]=%b",
  //    idx,
  //    dut.thermo_bus[idx*core_pkg::BITS_PER_OBS +: core_pkg::BITS_PER_OBS]);
  //endtask

  //task automatic print_thermo_bus();
  //  $display("thermo_bus (each %0d bits):", core_pkg::BITS_PER_OBS);
  //  for (int i = 0; i < SHOW_OBS; i++) print_thermo_block(i);
  //endtask

  //task automatic print_thermo_bus_clean();
    // prints the entire packed vector as-is
  //  $display("thermo_bus[%0d] = %b", $bits(dut.thermo_bus), dut.thermo_bus);
    // (optional) also show the registered version
    // $display("thermo_bus_r
  //endtask

  task automatic print_luts_mapped_to_layer0();
    $display("luts_mapped_to_layer0 (first %0d of %0d):", SHOW_LUTS, core_pkg::LUTS);
    for (int l = 0; l < SHOW_LUTS; l++)
      $display("  map0[%0d]=%b", l, dut.luts_mapped_to_layer0[l]);
  endtask

  task automatic print_internal_bits_layer0();
    $display("internal_bits_layer0 = %b", dut.internal_bits_layer0);
  endtask

  // print once per cycle
  always @(posedge clk) begin
    #1;
    $display("\n[%0t] ---- cycle debug ----", $time);
    // print_thermo_bus_clean();
    print_luts_mapped_to_layer0();
    print_internal_bits_layer0();
      $display("[DBG] in_reg: %0d %0d", int'(dut.in_reg[0]), int'(dut.in_reg[1]));
      $display("[%0t] in_reg[0]=%0d in_reg[1]=%0d  (X=%0d bits)",
    $time, int'(dut.in_reg[0]), int'(dut.in_reg[1]), $bits(dut.in_reg[0]));
    //$display("thermo_bus     [%0d] = %b", $bits(dut.thermo_bus), dut.thermo_bus);
    //$display("thermo_bus_r   [%0d] = %b", $bits(dut.thermo_bus_r), dut.thermo_bus_r);
    //$display("thermo_inside   [%0d] = %b", $bits(dut.thermo_bus_r), dut.thermo_bus_r);
    // Also show each block explicitly
  end
  // ---------- DEBUG: to here ----------
  // ------------------------------------------------------------------//
  //  TB variables (declare BEFORE initial block)
  // ------------------------------------------------------------------//
  string  IN_FILE  = "/nfs/scistore16/tomgrp/fkresse/difflogic/release_dwc/hdl/verilog/pkg/HalfCheetah/128/inputs.txt";
  string  OUT_FILE = "/nfs/scistore16/tomgrp/fkresse/difflogic/release_dwc/hdl/verilog/pkg/HalfCheetah/128/outputs.txt";

  integer fin, fout;
  integer eof_in, eof_out;
  integer line = 0;

  // reusable variables
  int rv;
  int exp [0:ACTIONS-1];   // expected outputs (from file)
  int errors;
  int actual;              // temp for comparisons

  // pattern array is same shape/type as obs, so we can assign directly
  logic [0:core_pkg::ENCODE_OBS_BIT_WIDTH-1] pattern [0:core_pkg::OBS-1];

  // ------------------------------------------------------------------//
  //  Stimulus + checker
  // ------------------------------------------------------------------//
  initial begin
    // open files
    fin  = $fopen(IN_FILE, "r");
    if (fin == 0) $fatal(1, "Could not open %s", IN_FILE);

    fout = $fopen(OUT_FILE, "r");
    if (fout == 0) $fatal(1, "Could not open %s", OUT_FILE);

    // main loop
    forever begin
      // 1) read one line of OBS integers into obs[]
      eof_in = 0;
      for (int oi = 0; oi < core_pkg::OBS; oi++) begin
        int tmp;
        rv = $fscanf(fin, "%d", tmp);
        if (rv != 1) begin
          eof_in = 1;
          break;
        end
        obs[oi] = tmp;  // assign explicitly to the packed vector
      end
      if (eof_in) break;

      // (optional) echo what we just read
      $write("[READ] ");
      for (int oi = 0; oi < core_pkg::OBS; oi++) $write("%0d ", int'(obs[oi]));
      $write("\n");

      // 2) wait pipeline latency
      @(posedge clk);
      repeat (PIPE_LAT) @(posedge clk);
      #1;

      // 3) read expected ACTIONS integers
      eof_out = 0;
      for (int a = 0; a < ACTIONS; a++) begin
        rv = $fscanf(fout, "%d", exp[a]);
        if (rv != 1) begin
          eof_out = 1;
          break;
        end
      end
      if (eof_out) break;

      // 4) compare / report
      errors = 0;
      for (int a = 0; a < ACTIONS; a++) begin
        actual = act[a]; // pack into int
        if (actual !== exp[a]) begin
          $display("[%0t] line %0d: action[%0d] MISMATCH  got=%0d  exp=%0d",
                   $time, line, a, actual, exp[a]);
          errors++;
        end
      end

      if (errors == 0)
        $display("[%0t] line %0d: PASS  outputs=%p", $time, line, act);

      line++;
    end

    $fclose(fin);
    $fclose(fout);
    $display("Completed %0d vectors.", line);
    $finish;
  end

endmodule
