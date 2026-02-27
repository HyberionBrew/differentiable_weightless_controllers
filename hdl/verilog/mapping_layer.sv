`include "core_pkg.svh"
`timescale 1ns/1ps

module dwn_mapping #(
    parameter int BITS_IN_W   = core_pkg::OBS * core_pkg::BITS_PER_OBS,
    parameter int OUT_LUTS  = core_pkg::LUTS,
    parameter int unsigned MAPPING [0:OUT_LUTS*core_pkg::INPUTS_LUT-1]
             = core_pkg::MAPPING_0
)(
    input logic [0:BITS_IN_W-1] bits_in,   // flat vector
    output logic [0:core_pkg::INPUTS_LUT-1] luts_input [0:OUT_LUTS-1]       // one output per LUT
);

genvar j, k;
    generate
        for (j = 0; j < OUT_LUTS; j++) begin : LUT_ASSIGNMENT
            // Collect I selected bits for this LUT
            for (k = 0; k < core_pkg::INPUTS_LUT; k++) begin : MAP
                assign luts_input[j][k] = bits_in[MAPPING[j*core_pkg::INPUTS_LUT + k]];
            end
        end
    endgenerate
endmodule
