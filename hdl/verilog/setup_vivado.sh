# 1. Prepend the system crt directory so ld can find crt1.o, crti.o, etc.
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}

# 2. Source Vivado settings (adjust the path if you installed elsewhere).
source ~/tools/Xilinx/Vivado/2022.2/settings64.sh

# if user command, run it; else drop into an interactive shell
if [[ $# -gt 0 ]]; then
    exec "$@"
else
    echo "Vivado environment ready.  Type your Vivado/xsim commands."
    exec "$SHELL"
fi
