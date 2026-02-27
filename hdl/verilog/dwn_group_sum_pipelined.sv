module dwn_group_sum #(
    parameter int BITS_IN     = core_pkg::LUTS,
    parameter int K           = core_pkg::ACTIONS,
    parameter int PIPE_STAGES = 1  // 0 = Combinational (Bad), 1+ = Pipelined (Good)
)(
    input  wire                      clk,          // NEW: Clock is required now!
    input  logic [0:BITS_IN-1]       bits_in,
    output logic [0:core_pkg::ACTION_BITWIDTH-1] group_sum [0:K-1]
);

    // ----------------------------------------------------------------
    // Group sizing with zero-padding
    // ----------------------------------------------------------------
    localparam int G   = (BITS_IN + K - 1) / K;
    localparam int PAD = G*K - BITS_IN;
    localparam int W   = $clog2(G+1);

    // ----------------------------------------------------------------
    // PIPELINING CONFIGURATION
    // ----------------------------------------------------------------
    // We split the large G vector into smaller chunks to break the adder tree.
    // 128 bits is a "sweet spot" for Artix-7 (fits in one slice column roughly).
    localparam int CHUNK_SIZE = 128;
    localparam int NUM_CHUNKS = (G + CHUNK_SIZE - 1) / CHUNK_SIZE;
    localparam int CHUNK_W    = $clog2(CHUNK_SIZE+1);

    genvar g, c, b;

    generate
        for (g = 0; g < K; g++) begin : GROUP
            
            // 1. Extract the full slice for this group
            logic [0:G-1] slice;
            for (b = 0; b < G; b++) begin : SLICE_MAP
                localparam int SRC = g*G + b;
                if (SRC < BITS_IN) assign slice[b] = bits_in[SRC];
                else               assign slice[b] = 1'b0;
            end

            if (PIPE_STAGES == 0) begin : GEN_COMB
                // ------------------------------------------------------------
                // Original Combinational Logic (High Delay)
                // ------------------------------------------------------------
                assign group_sum[g] = popcount(slice);
            
            end else begin : GEN_PIPE
                // ------------------------------------------------------------
                // Pipelined "Divide and Conquer"
                // ------------------------------------------------------------
                
                // Array to hold partial sums for each chunk
                logic [CHUNK_W-1:0] partial_sums [NUM_CHUNKS-1:0];
                
                // Step A: Combinational Popcount of small chunks
                for (c = 0; c < NUM_CHUNKS; c++) begin : CHUNKS
                    // Determine bounds for this chunk
                    localparam int LOW  = c * CHUNK_SIZE;
                    localparam int HIGH = (c+1) * CHUNK_SIZE - 1;
                    localparam int ACT_HIGH = (HIGH < G) ? HIGH : G-1;
                    localparam int WIDTH = ACT_HIGH - LOW + 1;

                    if (WIDTH > 0) begin
                        assign partial_sums[c] = popcount_chunk(slice[LOW : LOW+WIDTH-1]);
                    end else begin
                        assign partial_sums[c] = '0;
                    end
                end

                // Step B: Register the partial sums (Pipeline Stage 1)
                // This breaks the critical path: we only sum ~128 bits, then stop.
                reg [CHUNK_W-1:0] partial_regs [NUM_CHUNKS-1:0];
                always_ff @(posedge clk) begin
                    partial_regs <= partial_sums;
                end

                // Step C: Sum the registered partials
                // Summing ~80 integers of 8-bits is much faster than summing 10,000 bits.
                logic [W-1:0] total_sum_comb;
                always_comb begin
                    total_sum_comb = '0;
                    for (int i = 0; i < NUM_CHUNKS; i++) begin
                        total_sum_comb += partial_regs[i];
                    end
                end

                // Step D: Optional Final Register (Pipeline Stage 2)
                if (PIPE_STAGES >= 2) begin : STAGE_2
                    reg [W-1:0] final_reg;
                    always_ff @(posedge clk) final_reg <= total_sum_comb;
                    assign group_sum[g] = final_reg;
                end else begin : NO_STAGE_2
                    assign group_sum[g] = total_sum_comb;
                end

            end // end GEN_PIPE
        end // end GROUP loop
    endgenerate

    // ----------------------------------------------------------------
    // Helper Functions
    // ----------------------------------------------------------------
    
    // Original Main Popcount (for full width)
    function automatic [0:W-1] popcount (input logic [0:G-1] v);
        integer i;
        begin
            popcount = '0;
            for (i = 0; i < G; i++) popcount += v[i];
        end
    endfunction

    // Smaller Popcount (for chunks) - needed because of width mismatch
    function automatic [CHUNK_W-1:0] popcount_chunk (input logic [0:CHUNK_SIZE-1] v);
        integer i;
        begin
            popcount_chunk = '0;
            // Note: SystemVerilog functions handle variable input width 
            // implicitly if mapped correctly, but loop limit must be safe.
            for (i = 0; i < CHUNK_SIZE; i++) begin 
                 // We rely on 'v' being padded with 0s if input was smaller in the call
                 if (i < $bits(v)) popcount_chunk += v[i];
            end
        end
    endfunction

endmodule