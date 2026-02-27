#!/usr/bin/env python3
"""
export_model.py

Load a Stable-Baselines3 .zip file and print the `student` sub-module
of the actor network:
    python export_model.py --path=/path/to/your/model.zip
"""

import argparse
from thermometer import *
import math
import torch

import numpy as np
from wnn_models import WNN
import re
from pathlib import Path

def export_to_verilog(layers, obs_dim, act_dim, bits, filename="lut_init.sv"):
    """
    Export model[0].luts (shape [LUTS, 2**k]) into a SystemVerilog package.
    
    Parameters
    ----------
    model : torch.nn.Sequential
        Your network whose first layer is dwn.LUTLayer.
    filename : str or Path
        Path of the .sv file to create.
    """
    # 1. grab tensor on CPU, make sure it's ints 0 / 1
    # get first luts
    
    luts = layers[0].luts.detach().cpu().round().to(torch.int)
    print(luts)
    luts_layers, cols = luts.shape
    k = int(math.log2(cols))
    assert cols == (1 << k), "Second dimension must be a power of two"

    # 2. start building the SV source as a list of strings
    lines = []
    lines.append("`ifndef CORE_PKG_SVH")
    lines.append("`define CORE_PKG_SVH\n")
    lines.append("`timescale 1ns/1ps")
    lines.append("package core_pkg;\n")
    lines.append(f"  parameter int OBS = {obs_dim};")
    lines.append(f"  parameter int LUTS       = {luts_layers};")
    lines.append(f"  parameter int INPUTS_LUT = {k};\n")
    lines.append(f"  parameter int BITS_PER_OBS = {bits};")
    lines.append(f"  parameter int N_BITS = INPUTS_LUT * LUTS;")
    lines.append(f"  parameter int ACTIONS = {act_dim};")
    luts_last_layer = layers[-1].luts.shape[0]
    lines.append(f"  parameter int LAST_LUTS = {luts_last_layer};")
    lines.append(f"  parameter int ACTION_BITWIDTH = $clog2(LAST_LUTS/ACTIONS+1); // bitwidth of action outputs")
    lines.append(f"  parameter int ENCODE_OBS_BIT_WIDTH = $clog2(BITS_PER_OBS+1); // bitwidth for encoding observation inputs")
    # Last layer LUTS


    for l, layer in enumerate(layers):
        lines.append("  // Mapping array")
        if l == len(layers) -1:
            luts_n = luts_last_layer
            luts_str = 'LAST_LUTS'
        else:
            luts_n = luts_layers
            luts_str = 'LUTS'
            #lines.append(f"  parameter int unsigned MAPPING_{l} [0:LAST_LUTS* INPUTS_LUT-1] = '")
        lines.append(f"  parameter int unsigned MAPPING_{l} [0:{luts_str}* INPUTS_LUT-1] = '")
        lines[-1] += "{"
        luts = (layer.luts.detach().cpu() > 0.0).to(torch.int)
        if hasattr(layer.mapping, 'weights'):
            mapping = layer.mapping.weights.argmax(dim=0).cpu().detach().numpy().reshape(-1, k)
        else:
            print(layer.mapping.cpu().detach().numpy())
            print(layer.mapping.cpu().detach().numpy().shape)
            mapping = layer.mapping.cpu().detach().numpy().reshape(-1, k)
        for i in range(luts_n):
        # do it from top to bottom to have the right indexing starting with top
        # for i in range(LUTS-1, -1, -1):
            mapping_row = mapping[i] #[::-1] # have to invert the mapping to comforme 
            
            bits = ", ".join(f"{b.item()}" for b in mapping_row)
            comma = "," if i < luts_n - 1 else ""  # no trailing comma
            #comma = "," if i > 0 else ""        # no trailing comma on last
            lines.append(f"    {bits}{comma}   // LUT {i}")
        lines.append("  };")
        
        
        lines.append(f"  parameter logic [0:{luts_str}-1][0:(1<<INPUTS_LUT)-1] LUT_INIT_{l} = '")
        lines[-1] += "{"
        # 3. one sub‑array per LUT        #for idx, row in enumerate(luts):
        # reverse the order of rows to have LUT 0 at the top
        print(luts.shape)
        lut_lines = []
        for idx, row in enumerate(luts):
            # reverse tensor
            row = reversed(row)  # reverse the order of bits in each LUT
            bits = ", ".join(f"1'b{b.item()}" for b in row)
            comma = "," if idx < luts_n - 1 else ""        # no trailing comma on last
            #comma = "," if idx > 0 else ""        # no trailing comma on first
            lut_lines.append(f"    '{{{bits}}}{comma}   // LUT {idx}")
        lut_lines = lut_lines #[::-1]
        lines.extend(lut_lines)
        lines.append("  };")
        lines.append("\n")  
    lines.append("\nendpackage\n`endif\n")

    # 4. write to disk
    Path(filename).write_text("\n".join(lines))
    print(f"Wrote {filename} with {luts_n} LUTs × 2^{k} entries each.")

def make_group_sum_hook(tag: str = "GROUPSUM", 
                         store_attr: str = "_dbg_last_groupsum_out"):
    """
    Hook that prints the output of a GroupSum layer and stores it on the module
    as `module.<store_attr>` (CPU tensor) for later programmatic checks.
    """
    def _hook(module, inp, out=None):
        x = inp[0] if out is None else out  # pre-hook vs forward-hook
        out = x.detach()

        # Print a compact summary
        batch_like = out.view(-1, out.size(-1))  # flatten leading dims for printing
        to_show = min(4, batch_like.size(0))
        print(f"\n[{tag}] groupsum output  shape={tuple(out.shape)}  device={x.device}")
        print(batch_like[:to_show].cpu())

        # Optional: keep a copy for later checks
        if store_attr:
            try:
                setattr(module, store_attr, out.to("cpu"))
            except Exception as e:
                print(f"[{tag}] WARNING: could not store attr '{store_attr}': {e}")
        return

    return _hook
def make_thermo_values_hook(obs_dim: int, bits_per_obs: int, tag: str = "THERMO", 
                            store_attr: str = "_dbg_last_thermo_vals"):
    """
    Reconstructs obs_dim integers from a flattened thermometer code of length obs_dim*bits_per_obs
    along the last dimension. Works as forward pre-hook or forward hook.

    - Assumes grouping like: for i in [0..obs_dim-1], block i is j in [0..bits_per_obs-1],
      i.e., contiguous blocks per observation (matches thermo_out[i*BITS_PER_OBS + j]).
    - Coerces to {0,1} before summation: (>0.5) for float, (!=0) for integer/bool.

    The hook prints the recovered integers and also stores them on the module as
    `module.<store_attr>` (CPU tensor) for later programmatic checks.
    """
    def _hook(module, inp, out=None):
        x = inp[0] if out is None else out  # pre-hook vs forward-hook
        flat = x.detach()

        # Basic shape checks
        if flat.size(-1) != obs_dim * bits_per_obs:
            print(f"[{tag}] ERROR: last dim {flat.size(-1)} != obs_dim*bits_per_obs "
                  f"({obs_dim}*{bits_per_obs}={obs_dim*bits_per_obs})")
            return  # do not modify output

        # Reshape to (..., obs_dim, bits_per_obs)
        resh = flat.view(*flat.shape[:-1], obs_dim, bits_per_obs)

        # Force binary (thermometer should be 0/1)
        if resh.is_floating_point():
            bits01 = (resh > 0.5)          # float safety
        else:
            bits01 = (resh != 0)           # int/bool safety

        # Sum along thermometer axis -> integers per observation
        vals = bits01.sum(dim=-1)          # shape (..., obs_dim), dtype=torch.int64 (bool->long)

        # Optional: keep a copy for later checks
        if store_attr:
            try:
                setattr(module, store_attr, vals.to("cpu"))
            except Exception as e:
                print(f"[{tag}] WARNING: could not store attr '{store_attr}': {e}")

        # Print a compact summary
        batch_like = vals.view(-1, obs_dim)  # flatten leading dims for printing
        to_show = min(4, batch_like.size(0))
        print(f"\n[{tag}] thermo→values  shape={tuple(vals.shape)}  device={x.device}")
        print(batch_like[:to_show].cpu())

        # Do NOT return anything—returning from a forward hook would replace the module's output.
        return

    return _hook

def main():
    parser = argparse.ArgumentParser(
        description="Load a model and write hdl file"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the model file",
    )
    parser.add_argument(
        "--out",
        required=False,
        default="hdl/verilog/pkg",
        help="Path to the output .sv file",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=63,
        help="Number of bits for thermometer encoding",
    )
    parser.add_argument(
        "--random_interconnect",
        action='store_true',
        help="Use random interconnect mapping instead of learned mapping for SECOND layer",
    )
    
    args = parser.parse_args()
    # create the out path if it does not exist

    Path(args.out).mkdir(parents=True, exist_ok=True)
    #extract the name of the model from the path
    env_name = re.search(r'_([A-Za-z0-9]+)-v4', args.path).group(1)
    Path(f"{args.out}/{env_name}").mkdir(parents=True, exist_ok=True)
 
    # torch load path 
    print(f"Loading model from {args.path}")
    model = torch.load(args.path, map_location="cpu")
    print(model['fc_mean'].keys())
    obs_dim = model['obs_norm']['mean'].shape[0]
    
    beta_key = next(k for k in model['fc_mean'].keys() if 'beta' in k)
    act_dim = model['fc_mean'][beta_key].shape[0]
    print(act_dim)
    size=model['fc_mean']['net.1.luts'].shape[0]
    Path(f"{args.out}/{env_name}/{size}").mkdir(parents=True, exist_ok=True)
 
    final_path=f"{args.out}/{env_name}/{size}"
    # hard coded layer number
    l=2
    bits = args.bits
    thermo = ThermometerGaussian(n_bits=bits, device='cuda')
    min_values = torch.ones((obs_dim,)) * -10
    max_values = torch.ones((obs_dim,)) * 10
    thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)

    model =WNN(obs_dim=obs_dim, 
                act_dim=act_dim, 
                sizes=[size] * l, 
                thermometer=thermo, bits=bits,
                later_learnable=not(args.random_interconnect)) 
   
    raw_dict = torch.load(args.path, map_location="cpu")['fc_mean']
    print(raw_dict.keys())
    model.load_state_dict(raw_dict)
    model_net = model.net
    export_to_verilog([model_net[1],model_net[2]], obs_dim, act_dim, bits, filename=f"{final_path}/core_pkg.svh")


    model_net[1].register_forward_pre_hook(make_thermo_values_hook(obs_dim, bits))
    model_net[3].register_forward_hook(make_group_sum_hook())
    
    # pass random randn through
    x = torch.randn((1, obs_dim)).cuda()
    model = model.cuda()
    out = model(x)
    
    
    inputs = model_net[1]._dbg_last_thermo_vals
    outputs = model_net[3]._dbg_last_groupsum_out.type(torch.int32)
    inp_np = inputs.to(torch.int64).cpu().numpy()
    out_np = outputs.to(torch.int64).cpu().numpy()
    print(inp_np)
    print(out_np)
    np.savetxt(f"{final_path}/inputs.txt",  inp_np, fmt="%d")
    np.savetxt(f"{final_path}/outputs.txt", out_np, fmt="%d")

if __name__ == "__main__":
    main()
