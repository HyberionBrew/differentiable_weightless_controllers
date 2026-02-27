########################################################################
##  dwn_top – Artix‑7  xc7a200t‑2fbg484‑2   (DEBUG/ESTIMATE ONLY)
##  17 × 5‑bit observation inputs  (85 pins)
##   6 × 5‑bit action outputs      (30 pins)
########################################################################

#####################
# 1. System clock  (100 MHz, GCLK pin)
#####################
# set_property PACKAGE_PIN Y9   [get_ports clk]
# set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -name sys_clk -period 10.0 [get_ports clk]

#####################
# 2. Asynchronous reset‑n
#####################
# set_property PACKAGE_PIN Y11  [get_ports rstn]
# set_property IOSTANDARD LVCMOS33 [get_ports rstn]

#####################
# 3. Observation bus  (17 × 5 bits → 85 pins)
#    in_bus[group][bit]  — group 0 … 16, bit 0 … 4
#####################
# Group 0  (in_bus[0][0:4])


#####################
# 5. I/O standards  (apply LVCMOS33 everywhere)
#####################
#####################
# 6. Downgrade “un‑LOC’d I/O” errors to warnings (keeps flow alive)
#####################
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
