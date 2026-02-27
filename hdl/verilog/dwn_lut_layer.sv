`include "core_pkg.svh"
`timescale 1ns/1ps

module dwn_lut_layer #(
        parameter int LUTS_CURR = core_pkg::LUTS,
        parameter logic [0:LUTS_CURR-1][0:(1<<core_pkg::INPUTS_LUT)-1] LUT_INIT
             = core_pkg::LUT_INIT_0
)
(
    input logic [0:core_pkg::INPUTS_LUT-1] lut_in [0:LUTS_CURR-1],   // flat vector
    output logic [0:LUTS_CURR-1] bits_out       // one output per LUT
);
    //-----------------------------------------------------------------
    // I/O
    //-----------------------------------------------------------------
    // input  logic [N_BITS-1:0] bits_in;     // flat vector
    // output logic [L-1:0]      bits_out;    // one output per LUT

    //-----------------------------------------------------------------
    // Remap + per‑LUT logic
    //-----------------------------------------------------------------
    genvar j;
    generate
        import core_pkg::*;
        for (j = 0; j < LUTS_CURR; j++) begin : LUT_BANK
            if (INPUTS_LUT == 6) begin
                LUT6 #(.INIT(LUT_INIT[j])) lut6_inst (
                    .I0(lut_in[j][0]), .I1(lut_in[j][1]), .I2(lut_in[j][2]),
                    .I3(lut_in[j][3]), .I4(lut_in[j][4]), .I5(lut_in[j][5]),
                    .O (bits_out[j])
                );
            end else if (INPUTS_LUT == 5) begin
                LUT5 #(.INIT(LUT_INIT[j][0:31])) lut5_inst (  // 2**5 = 32 bits
                    .I0(lut_in[j][0]), .I1(lut_in[j][1]), .I2(lut_in[j][2]),
                    .I3(lut_in[j][3]), .I4(lut_in[j][4]),
                    .O (bits_out[j])
                );
            end else if (INPUTS_LUT == 4) begin
                LUT4 #(.INIT(LUT_INIT[j][0:15])) lut4_inst (
                    .I0(lut_in[j][0]), .I1(lut_in[j][1]),
                    .I2(lut_in[j][2]), .I3(lut_in[j][3]),
                    .O (bits_out[j])
                );
            end else begin
                // Fallback: XOR‑reduce for small I (I <= 3) – safe placeholder
                assign bits_out[j] = ^lut_in[j];
            end
        end
    endgenerate
endmodule
