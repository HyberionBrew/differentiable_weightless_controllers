from typing import Optional, List, Union
import torch
import torch.nn as nn


from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity, QuantTanh
from dependencies import value

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver

from brevitas.quant.scaled_int import Int8Bias, Int32Bias

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# import quantization status manager
from brevitas.graph.calibrate import quantization_status_manager, calibration_mode

class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0
    
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import BitWidthImplType, ScalingImplType, QuantType, RestrictValueType
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from dependencies import value
from brevitas.core.stats.stats_op import AbsMax  # or MSE
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QuantizedModel(nn.Module):
    def __init__(self, obs_actor, act, hidden, activation_fn, 
                 act_bit_width=1, weight_bit_width=1, last_bit_width=8,
                 thermometer=None,squash=False, 
                 initial_quantization=None, is_ppo= False, no_fp=True,
                 ptq_calibration=True,
                 ):
        super(QuantizedModel, self).__init__()
        self.thermometer = thermometer
        print(thermometer)
        if thermometer is not None:
            obs_actor = obs_actor * self.thermometer.n_bits
        else:
            obs_actor = obs_actor
        print(f"obs_actor: {obs_actor}, act: {act}, hidden: {hidden}, act_bit_width: {act_bit_width}, weight_bit_width: {weight_bit_width}")
        print(last_bit_width, initial_quantization)
        print('-----------')
        self.quant_enabled = False
        self.no_fp = no_fp
        self.fp_model = None
        if act_bit_width is None:
            self.quant_model = None
            print("Create network somewhere else")
        else:
            assert weight_bit_width > 1, "Weight bitwidth less than 1 not supported"
            print('using quantization with weight bit width > 1')
            print('inital',initial_quantization)
            print('act', act_bit_width)
            print('last_bit', last_bit_width)
            print('is ppo', is_ppo)
            print('ptq_calibration', ptq_calibration)
            if is_ppo:
                # For PPO, we use the same layer init as in the standard PPO code.
                self.quant_model = nn.Sequential(
                    QuantIdentity(bit_width=initial_quantization, return_quant_tensor=True),
                    layer_init(QuantLinear(obs_actor, hidden[0],  bias=True,weight_bit_width=weight_bit_width, bias_quant=Int32Bias, )),
                    activation_fn(bit_width=act_bit_width, return_quant_tensor=True),
                    layer_init(QuantLinear(hidden[0], hidden[1],  bias=True,weight_bit_width=weight_bit_width, bias_quant=Int32Bias,)), 
                    activation_fn(bit_width=act_bit_width, return_quant_tensor=True), 
                    layer_init(QuantLinear(hidden[1], act,  bias=True,weight_bit_width=weight_bit_width, bias_quant=Int32Bias,), std=0.01),
                    QuantTanh(bit_width=last_bit_width, return_quant_tensor=True) if squash else QuantIdentity(bit_width=last_bit_width, return_quant_tensor=True),
                )
            else:
                self.quant_model = nn.Sequential(
                    QuantIdentity(bit_width=initial_quantization, return_quant_tensor=True),
                    QuantLinear(obs_actor, hidden[0],  bias=True,weight_bit_width=weight_bit_width, bias_quant=Int32Bias, ),
                    activation_fn(bit_width=act_bit_width, return_quant_tensor=True),
                    QuantLinear(hidden[0], hidden[1],  bias=True,weight_bit_width=weight_bit_width, bias_quant=Int32Bias,),
                    activation_fn(bit_width=act_bit_width, return_quant_tensor=True), 
                    QuantLinear(hidden[1], act,  bias=True,weight_bit_width=weight_bit_width, bias_quant=Int32Bias,),
                    QuantTanh(bit_width=last_bit_width, return_quant_tensor=True) if squash else QuantIdentity(bit_width=last_bit_width, return_quant_tensor=True),
                )

                
            self.warmup_ctx = quantization_status_manager(
                self.quant_model,
               disable_act_quant  = True,   # disable activation quant
               disable_weight_quant = True, # disable weights
               disable_bias_quant = True,   # usually want this too
               call_act_quantizer_impl = True, # collect statistics
               is_training=True)            # keep .training flag consistent
            

    def clip_weights(self, min_val=-1.0, max_val=1.0):
        """
        Clip the weights of the model to the specified range.
        """
        for m in self.modules():
            if isinstance(m, QuantLinear):
                m.weight.data.clamp_(min_val, max_val)
                print(f"Clipped weights of {m.__class__.__name__} to range [{min_val}, {max_val}]")
                
    def forward(self, x, deterministic: bool = True):
        """
        Forward pass through the quantized model.
        """
        
        if self.quant_enabled:
            out = self.quant_model(x)
            return out
        else:
            with self.warmup_ctx:
                out = self.quant_model(x)
                return out
        
    def disable_quant(self):
        """
        Disable quantization for the model.
        """
        self.quant_enabled = False
    def enable_quant(self):
        """
        Enable quantization for the model.
        """
        self.quant_enabled = True
        # copy weights from fp model to quant model
        if self.fp_model is None:
            print("No fp model to copy weights from.")
            return
        
        x = self.quant_model.load_state_dict(self.fp_model.state_dict(), strict=False)
        
    def set_training_mode(self, mode: bool = True):
        """
        Set the training mode for the model.
        """
        self.training = mode
        self.quant_model.train(mode)
    def get_std(self):
        raise NotImplementedError("QuantizedModel does not support get_std method. Use the quantized actor directly for actions.")
    
    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        raise NotImplementedError(msg)
