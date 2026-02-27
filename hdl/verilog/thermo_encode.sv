`timescale 1ns/1ps
`include "core_pkg.svh"

module thermo_encode #(
  // imported parameters
  parameter int OBS        = core_pkg::OBS,            // e.g. 17
  parameter int BITS_PER_OBS  = core_pkg::BITS_PER_OBS,      // e.g. 21
  // auto‑computed
  parameter int X          = core_pkg::ENCODE_OBS_BIT_WIDTH       // bits to encode 0…MAX_VALUE‑1
)(
  // a single-cycle wide bus carrying all quantized inputs
  input  logic [0:core_pkg::ENCODE_OBS_BIT_WIDTH-1] quant_in [0:core_pkg::OBS-1],   
  // the thermometer‑encoded bus, valid same cycle
  output logic [0:core_pkg::OBS*core_pkg::BITS_PER_OBS-1] thermo_out  
);

  // -------------------------------------------------------------------
  // Combinational thermometer expansion
  // thermo[i][j] = 1 if quant_in[i]>j, else 0
  // -------------------------------------------------------------------
  genvar i, j;
  generate
    for (i = 0; i < OBS; i++) begin : G_OBS
      for (j = 0; j < BITS_PER_OBS; j++) begin : G_BIT
        // quant_in[i] is already an X‑bit packed vector
        // j is an int literal; comparison is legal
        assign thermo_out[i*BITS_PER_OBS + j] = (quant_in[i] > j);
      end
    end
  endgenerate
  // Optional register stage if you want to meet timing
  // always_ff @(posedge clk) if (!rstn) thermo_out <= '0; else thermo_out <= thermo_out_comb;

endmodule
