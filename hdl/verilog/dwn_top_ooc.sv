// dbn_top.sv  – tie thresholds + DBN core together
`include "core_pkg.svh"
`timescale 1ns/1ps


module dwn_top #(
    parameter int PIPE_STAGES = 4  // 0, 1, or 2 internal pipeline stages
)(
    input  logic clk,
    input  logic rstn,
    // input  logic [0:core_pkg::ENCODE_OBS_BIT_WIDTH-1] in_bus [0:core_pkg::OBS-1],
    input logic [0:core_pkg::OBS*core_pkg::BITS_PER_OBS-1] in_bus,
    output logic [0:core_pkg::ACTION_BITWIDTH-1] group_sum [0:core_pkg::ACTIONS-1]
);
    import core_pkg::*;

    // ── Front‑end: comparators ───────────────────────────────────
    // ── Boolean policy network ──────────────────────────────────

    logic [0:INPUTS_LUT-1] luts_mapped_to_layer0 [0:LUTS-1]; // LUT inputs
    logic [0:INPUTS_LUT-1] luts_mapped_to_layer0_pipe [0:LUTS-1];

    logic [0:INPUTS_LUT-1] luts_mapped_to_layer1 [0:LAST_LUTS-1]; // LUT inputs
    logic [0:INPUTS_LUT-1] luts_mapped_to_layer1_pipe [0:LAST_LUTS-1];
    logic [0:LUTS-1] internal_bits_layer0; // LUT outputs
    logic [0:LUTS-1]       internal_bits_layer0_pipe;  // after optional pipe

    logic [0:LAST_LUTS-1] output_bits; // LUT outputs
    logic [0:LAST_LUTS-1] output_bits_r; // LUT outputs
    logic [0:core_pkg::OBS*core_pkg::BITS_PER_OBS-1] thermo_bus;

    // logic [0:LUTS-1] internal_bits_layer0_r; // existing
    // logic [0:LUTS-1] output_bits_r;                              // NEW FF
    logic [0:LAST_LUTS-1]  output_bits_pipe;   // after optional pipe
    
    logic [0:INPUTS_LUT-1] luts_mapped_to_layer0_r    [0:LUTS-1]; // NEW FF
    logic [0:INPUTS_LUT-1] luts_mapped_to_layer1_r    [0:LAST_LUTS-1]; // NEW FF
    // logic [0:LUTS-1] output_bits_r;                              // NEW FF
    logic [0:OBS*BITS_PER_OBS-1] thermo_bus_r;   


    logic [0:ENCODE_OBS_BIT_WIDTH-1] in_reg [0:OBS-1];

    // INPUT REGISTER
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn)
            foreach (thermo_bus[i])  thermo_bus[i] <= '0;
        else
            foreach (thermo_bus[i])  thermo_bus[i] <= in_bus[i];
    end


    //#thermo_encode #(
    //.OBS       (OBS),
    //.BITS_PER_OBS(BITS_PER_OBS)
    //) u_enc (
    //    .quant_in   (in_reg),
    //    .thermo_out (thermo_bus)
    // );
    

    dwn_mapping #(
        .BITS_IN_W (OBS*BITS_PER_OBS),   // = 357
        .OUT_LUTS  (LUTS),
        .MAPPING(MAPPING_0)
        ) map0 (
        .bits_in (thermo_bus),
        .luts_input (luts_mapped_to_layer0)
    );
// Helps route long wires from input to first layer
    generate
        if (PIPE_STAGES > 2) begin : gen_map0_pipe
            logic [0:INPUTS_LUT-1] luts_mapped_to_layer0_r [0:LUTS-1];
            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn) foreach(luts_mapped_to_layer0_r[i]) luts_mapped_to_layer0_r[i] <= '0;
                else       foreach(luts_mapped_to_layer0_r[i]) luts_mapped_to_layer0_r[i] <= luts_mapped_to_layer0[i];
            end
            assign luts_mapped_to_layer0_pipe = luts_mapped_to_layer0_r;
        end else begin : gen_map0_direct
            assign luts_mapped_to_layer0_pipe = luts_mapped_to_layer0;
        end
    endgenerate

    dwn_lut_layer #(
        .LUT_INIT(LUT_INIT_0)
        ) 
        layer0 (
        .lut_in (luts_mapped_to_layer0_pipe),
        .bits_out(internal_bits_layer0)
    );
    // logic [0:LUTS-1] internal_bits_layer0_r;

    //always_ff @(posedge clk or negedge rstn) begin
    //    if (!rstn)
    //        internal_bits_layer0_r <= '0;
    //    else
    //        internal_bits_layer0_r <= internal_bits_layer0;
    //end

    generate
        if (PIPE_STAGES >= 1) begin : gen_pipe_stage_1
            logic [0:LUTS-1] internal_bits_layer0_r;
            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn)
                    internal_bits_layer0_r <= '0;
                else
                    internal_bits_layer0_r <= internal_bits_layer0;
            end
            assign internal_bits_layer0_pipe = internal_bits_layer0_r;
        end else begin : gen_no_pipe_stage_1
            // no pipeline: directly connect
            assign internal_bits_layer0_pipe = internal_bits_layer0;
        end
    endgenerate


    dwn_mapping #(
        .BITS_IN_W (LUTS),
        .OUT_LUTS  (LAST_LUTS),
        .MAPPING(MAPPING_1)
        ) map1 (
        .bits_in (internal_bits_layer0_pipe),
        .luts_input (luts_mapped_to_layer1)
    );

    generate
        if (PIPE_STAGES > 2) begin : gen_map1_pipe
            logic [0:INPUTS_LUT-1] luts_mapped_to_layer1_r [0:LAST_LUTS-1];
            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn) foreach(luts_mapped_to_layer1_r[i]) luts_mapped_to_layer1_r[i] <= '0;
                else       foreach(luts_mapped_to_layer1_r[i]) luts_mapped_to_layer1_r[i] <= luts_mapped_to_layer1[i];
            end
            assign luts_mapped_to_layer1_pipe = luts_mapped_to_layer1_r;
        end else begin : gen_map1_direct
            assign luts_mapped_to_layer1_pipe = luts_mapped_to_layer1;
        end
    endgenerate

    dwn_lut_layer#(
        .LUT_INIT(LUT_INIT_1),
        .LUTS_CURR (LAST_LUTS)
        ) 
        layer1 (
        .lut_in (luts_mapped_to_layer1_pipe),
        .bits_out(output_bits)
    );
    
    logic [0:ACTION_BITWIDTH-1] group_sum_int [0:ACTIONS-1];
    // add a stage here?
    //always_ff @(posedge clk or negedge rstn) begin
    //    if (!rstn)
    //        output_bits_r <= '0;
    //    else
    //        output_bits_r <= output_bits;
    //end
    generate
        if (PIPE_STAGES >= 2) begin : gen_pipe_stage_2
            logic [0:LAST_LUTS-1] output_bits_r;
            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn)
                    output_bits_r <= '0;
                else
                    output_bits_r <= output_bits;
            end
            assign output_bits_pipe = output_bits_r;
        end else begin : gen_no_pipe_stage_2
            assign output_bits_pipe = output_bits;
        end
    endgenerate

    localparam int GS_STAGES = (PIPE_STAGES > 2) ? (PIPE_STAGES - 2) : 0;

    dwn_group_sum #(
        .BITS_IN (LAST_LUTS),    // = number of LUT outputs
        .K       (ACTIONS),
        .PIPE_STAGES (GS_STAGES) // define GROUPS in core_pkg
    ) u_gs (
        .bits_in    (output_bits_pipe),  // <‑‑  from LUT layer
        .group_sum  (group_sum_int),
        .clk        (clk)
    );
    
    // ── Output register ──────────────────────────────────────────
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn)
            foreach (group_sum[i])  group_sum[i] <= '0;
        else
            foreach (group_sum[i])  group_sum[i] <= group_sum_int[i];
    end


endmodule
