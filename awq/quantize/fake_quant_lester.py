import torch
import torch.nn as nn
import torch.nn.functional as F
from awq.quantize.quantizer import pseudo_quantize_tensor
from typing import Literal
from functools import partial


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_n_bits: int = 4,
        a_n_bits: int = 4,
        act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
        quantize_output: bool = False,
    ):
        super().__init__()
        assert act_quant in ["per_token", "per_tensor", "none"]

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float32,
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features),
                    dtype=torch.float32,
                    requires_grad=False,
                ),
            )

        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=a_n_bits
            )
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=a_n_bits
            )
        else:
            self.act_quant_name = "None"
            self.act_quant = lambda x: x

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(QuantizedLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):

        x = self.act_quant(x)
        x = F.linear(x, self.weight, self.bias)
        x = self.output_quant(x)

        return x

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        w_n_bits: int = 4,
        a_n_bits: int = 4,
        zero_point: bool = True,
        group_size: int = 128,
        act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
        quantize_output: bool = False,
    ):

        awq_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )

        awq_linear.weight.data = pseudo_quantize_tensor(
            w=linear.weight.data,
            n_bit=w_n_bits,
            zero_point=zero_point,
            q_group_size=group_size,
        )

        if linear.bias is not None:
            awq_linear.bias.data = linear.bias.data

        return awq_linear


def quantize_opt_model(
    model,
    w_n_bits: int = 4,
    a_n_bits: int = 4,
    zero_point: bool = True,
    group_size: int = 128,
    act_quant: Literal["per_token", "per_tensor", "none"] = "none",
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTAttention) or isinstance(m, OPTDecoderLayer):
            model.model = quantize_opt_layer(
                m,
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
            )

    return model


def quantize_opt_layer(
    m: nn.Module,
    w_n_bits: int = 4,
    a_n_bits: int = 4,
    zero_point: bool = True,
    group_size: int = 128,
    act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
):
    from transformers.models.opt.modeling_opt import (
        OPTDecoderLayer,
        OPTAttention,
    )

    if isinstance(m, OPTDecoderLayer):
        m.fc1 = QuantizedLinear.from_linear(
            m.fc1,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            zero_point=zero_point,
            group_size=group_size,
            act_quant=act_quant,
        )
        m.fc2 = QuantizedLinear.from_linear(
            m.fc2,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            zero_point=zero_point,
            group_size=group_size,
            act_quant=act_quant,
        )
    elif isinstance(m, OPTAttention):
        # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
        m.q_proj = QuantizedLinear.from_linear(
            m.q_proj,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            zero_point=zero_point,
            group_size=group_size,
            act_quant=act_quant,
            quantize_output=True,
        )
        m.k_proj = QuantizedLinear.from_linear(
            m.k_proj,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            zero_point=zero_point,
            group_size=group_size,
            act_quant=act_quant,
            quantize_output=True,
        )
        m.v_proj = QuantizedLinear.from_linear(
            m.v_proj,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            zero_point=zero_point,
            group_size=group_size,
            act_quant=act_quant,
            quantize_output=True,
        )
        m.out_proj = QuantizedLinear.from_linear(
            m.out_proj,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            zero_point=zero_point,
            group_size=group_size,
            act_quant=act_quant,
            quantize_output=True,
        )

    return m
