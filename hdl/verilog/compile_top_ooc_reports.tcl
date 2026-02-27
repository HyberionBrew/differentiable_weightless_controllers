# ---------------------------------------------------------------------
# run_dwn_ooc.tcl – OOC synth / P&R for dwn_top 
# ---------------------------------------------------------------------
set env_name  "HalfCheetah"   ;# default env
set size      128          ;# default size
set pipe_regs 0              ;# default: 0 internal pipeline registers

foreach arg $argv {
    if {[regexp {^--env=(.+)} $arg -> v]} {
        set env_name $v
    } elseif {[regexp {^--size=(.+)} $arg -> v]} {
        set size $v
    } elseif {[regexp {^--pipe_regs=([0-9]+)} $arg -> v]} {
        set pipe_regs $v
    }
}

puts "Using env:        $env_name"
puts "Using size:       $size"
puts "Pipeline stages:  $pipe_regs"


set PKG_BASE "pkg/$env_name/$size"


# Read HDL sources 
read_verilog -sv "$PKG_BASE/core_pkg.svh"
read_verilog -sv dwn_lut_layer.sv
read_verilog -sv dwn_group_sum_pipelined.sv
read_verilog -sv dwn_top_ooc.sv
read_verilog -sv mapping_layer.sv
read_verilog -sv thermo_encode.sv


read_xdc dwn_top.xdc


set_param synth.elaboration.rodinMoreOptions "rt::set_parameter var_size_limit 4194304"


# larger: xc7a200tfbg484-1, xc7a15tfgg484
synth_design -top dwn_top -part xc7a15tfgg484-1 -mode out_of_context -generic "PIPE_STAGES=$pipe_regs"

opt_design

# -directive AltSpreadLogic_high
place_design
# phys_opt_design -directive AggressiveFanoutOpt
# phys_opt_design -directive AggressiveExplore

route_design
# Reports
set OUT_DIR "synth_results/$env_name/$size"
file mkdir $OUT_DIR

report_utilization       -file "$OUT_DIR/util.rpt"
report_methodology       -file "$OUT_DIR/methodology.rpt"
report_timing_summary    -file "$OUT_DIR/timing.rpt"  -max_paths 20
report_power             -file "$OUT_DIR/power.rpt"
# ----------------------------------------------------------
# Extract key metrics and write a short summary
# ----------------------------------------------------------

# 1) Get reports as strings (separate from the -file versions)
set util_str   [report_utilization      -return_string]
set timing_str [report_timing_summary   -return_string -max_paths 10]
set power_str  [report_power            -return_string]

# 2) Extract LUTs from "Slice LUTs" line
set num_luts "N/A"
if {[regexp {\|\s*Slice LUTs\s*\|\s*([0-9,]+)} $util_str -> luts]} {
    # remove thousands separators
    set num_luts [string map {"," ""} $luts]
}

# 3) Extract FFs from "Slice Registers" line
set num_ffs "N/A"
if {[regexp {\|\s*Slice Registers\s*\|\s*([0-9,]+)} $util_str -> ffs]} {
    set num_ffs [string map {"," ""} $ffs]
}

# 4) Extract WNS (Worst Negative Slack) from timing summary
set wns "N/A"
if {[regexp {WNS\(ns\)[^\n]*\n[- ]+\n\s*([-0-9\.]+)} $timing_str -> w]} {
    set wns $w
}

# Timing met if WNS >= 0
set timing_met "N/A"
if {$wns ne "N/A"} {
    if {$wns >= 0.0} {
        set timing_met "YES"
    } else {
        set timing_met "NO"
    }
}

# 5) Extract total on-chip power
set total_power "N/A"
if {[regexp {Total On-Chip Power\s*\(W\)\s*\|\s*([0-9\.]+)} $power_str -> p]} {
    set total_power $p
}


# 6) Write summary file
set OUT_DIR "synth_results/$env_name/$size"
file mkdir $OUT_DIR

set summary_file "$OUT_DIR/summary.txt"
set fh [open $summary_file w]

puts $fh "Env:          $env_name"
puts $fh "Size:         $size"
puts $fh ""
puts $fh "Slice LUTs:   $num_luts"
puts $fh "Registers:    $num_ffs"
puts $fh ""
puts $fh "WNS (ns):     $wns"
puts $fh "Timing met:   $timing_met"
puts $fh ""
puts $fh "Total power:  $total_power W"

close $fh

puts "Wrote summary to $summary_file"

# Save a design checkpoint for reuse / IP packaging
write_checkpoint -force dwn_top_ooc.dcp

quit
