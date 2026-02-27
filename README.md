This is the official code repository for the paper "Differentiable Weightless Controllers: Learning Logic Circuits for Continuous Control" (https://arxiv.org/abs/2512.01467).  

Code based on the CleanRL implementations of the corresponding algorithms: https://docs.cleanrl.dev/, and see License file.  

# Setup

1. Load system modules (if on a cluster, otherwise install locally, gcc 10 plays nice, higher versions might not work)

```bash
module load cuda/11.8
module load gcc/10
```

2. Install PyTorch with specific CUDA 11.8 support
```bash
pip install torch==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install DWN (Requires local clone)
```bash
git clone https://github.com/alanbacellar/DWN
```
```bash
pip install setuptools wheel
```
```bash
pip install -v --no-build-isolation ./DWN/
```

6. Install remaining dependencies
```bash
pip install -r requirements.txt
```

# Run examples
Run the training script by selecting a `network_type`. **Note:** `wnn` requires a CUDA-enabled device, while `float` and `quant` should be run on CPU.

### 1. Weightless Neural Network (WNN) - *Requires CUDA*
```bash
python sac.py --network-type wnn --l=2 --size 512 --bits 63 --n 6 --cuda --env-id=HalfCheetah-v4 --no-track
```


### 2. Float
```bash
python sac.py --network-type float --l 3 --size=256 --no-cuda --env-id=HalfCheetah-v4 --no-track
```

### 3. Quant 
```bash
python sac.py --network-type quant --l 3  --size=256 --n-bit-quantization 8 --initial-quantization 8 --last_bit_width 8 --no-cuda --env-id=HalfCheetah-v4 --no-track
```


## Overview Important Parameters

| Network Type | Relevant Arguments | Description |
| :--- | :--- | :--- |
| **Common** | `--l`, `--running-normalization`, `--save-path` | Layers, normalization, and model saving. |
| **wnn** | `--size`, `--bits`, `--n` | Input size, thermometer bits, and LUT inputs. |
| **quant** | `--n-bit-quantization`, `--initial-quantization`, `--last-bit-width`, `--enable-quant-step` | Bit-widths and quantization timing. |


## View Results

Results are per default saved in runs or models. They are saved in a tensorboard format:
```bash
tensorboard --logdir models
```

Results can also be logged to wandb.

# Verilog Conversion Code

## Setup 

Requires **Xilinx Vivado 2022.2** or newer. Update the paths in `setup_vivado.sh` to match your installation, then run:
```bash
chmod +x setup_vivado.sh
./setup_vivado.sh
```

## HDL Code
Convert a trained model to Verilog:

```bash
python export_hdl.py --path=PATH_TO_MODEL
```
This populates hdl/verilog/pkg with:

* .pkg: Network parameters (LUT contents, connections).
* Test Vectors: Input/output files for testing.

## Simulation
*Note*: Update `IN_FILE` and `OUT_FILE` in `tb_in_out.sv` (lines 92-93) to point to your generated test vectors before running.

Compile the design (ensure `-i` points to your specific package folder):
```bash
cd hdl/verilog
xvlog -sv   -i .   -i pkg/HalfCheetah/128   dwn_lut_layer.sv   dwn_group_sum_pipelined.sv   mapping_layer.sv   dwn_top_ooc.sv   thermo_encode.sv   tb_in_out.sv
```

Run simulation:
```bash
xelab tb_in_out -R -debug typical -L unisims_ver
```

This should return `pass`.

## Synthesis and Implementation
Run the batch synthesis flow for resource and timing reports (can be found in `hdl/verilog/synth_results`):

```bash
vivado -mode batch \
      -source compile_top_ooc_reports.tcl \
      -tclargs --env=HalfCheetah --size=128 --pipe_regs=1
```

* Use `--env` and `--size` to select the correct `.pkg` folder.

* Use `--pipe_regs` to tune the number of pipeline registers.