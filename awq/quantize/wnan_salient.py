import os
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

from .quantizer import pseudo_quantize_tensor


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    # t.shape = (input seq_len, hidden_size)

    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[
        0
    ]  # scales.shape = (input seq_len, 1) max along the channel dimension
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()  # scales.shape = (1) max along the entire tensor
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_channel_absmax(t, n_bits=8):
    # t.shape = (input seq_len, hidden_size)
    # input_feats = list of tensors of shape (hidden_size,) <- my guess is that len(input_feats) = input seq_len

    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=0, keepdim=True)[
        0
    ]  # scales.shape = (1, hidden_size) max along the channel dimension
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    return t


@torch.no_grad()
def quantize_activation_per_token_absmax_salient(t, outlier_indices, n_bits=8):
    # t.shape = (input seq_len, hidden_size)
    # input_feats = list of tensors of shape (hidden_size,) <- my guess is that len(input_feats) = input seq_len

    assert outlier_indices.dim() == 1  # shape = (1% of hidden_size,)

    t_copy = t.clone()

    t_shape = t.shape
    t.view(-1, t_shape[-1])

    scales = t.abs().max(dim=-1, keepdim=True)[
        0
    ]  # scales.shape = (input seq_len, 1) max along the channel dimension
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    t[..., outlier_indices] = t_copy[..., outlier_indices]

    return t


@torch.no_grad()
def quantize_activation_per_channel_absmax_salient(t, outlier_indices, n_bits=8):
    # t.shape = (input seq_len, hidden_size)
    # input_feats = list of tensors of shape (hidden_size,) <- my guess is that len(input_feats) = input seq_len

    assert outlier_indices.dim() == 1  # shape = (1% of hidden_size,)

    t_copy = t.clone()
    t_shape = t.shape
    t.view(-1, t_shape[-1])

    scales = t.abs().max(dim=0, keepdim=True)[
        0
    ]  # scales.shape = (1, hidden_size) max along the channel dimension

    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    t[..., outlier_indices] = t_copy[..., outlier_indices]

    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax_salient(t, outlier_indices, n_bits=8):
    # t.shape = (input seq_len, hidden_size)
    assert outlier_indices.dim() == 1  # shape = (1% of hidden_size,)

    t_copy = t.clone()

    t_shape = t.shape
    t.view(-1, t_shape[-1])

    scales = t.abs().max()  # scales.shape = (1) max along the entire tensor
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    t[..., outlier_indices] = t_copy[..., outlier_indices]

    return t


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_n_bits=4,
        a_n_bits=16,
        act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
        quantize_output: bool = False,
        outlier_indices: torch.Tensor = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
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
        elif act_quant == "per_channel":
            self.act_quant_name = "per_channel"
            self.act_quant = partial(
                quantize_activation_per_channel_absmax, n_bits=a_n_bits
            )
        elif act_quant == "per_token_salient":
            self.act_quant_name = "per_token_salient"
            self.act_quant = partial(
                quantize_activation_per_token_absmax_salient,
                outlier_indices=outlier_indices,
                n_bits=a_n_bits,
            )
        elif act_quant == "per_tensor_salient":
            self.act_quant_name = "per_tensor_salient"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax_salient,
                outlier_indices=outlier_indices,
                n_bits=a_n_bits,
            )
        elif act_quant == "per_channel_salient":
            self.act_quant_name = "per_channel_salient"
            self.act_quant = partial(
                quantize_activation_per_channel_absmax_salient,
                outlier_indices=outlier_indices,
                n_bits=a_n_bits,
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
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

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

        # this is a linear layer that will eventually enhouse the quantized weights
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

    @classmethod
    def from_linear_salient_weight(
        cls,
        linear: nn.Linear,
        input_feats: torch.Tensor,
        w_n_bits: int = 4,
        a_n_bits: int = 4,
        zero_point: bool = True,
        group_size: int = 128,
        act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
        quantize_output: bool = False,
        protection_ratio: float = 0.01,
    ):

        # this is a linear layer that will eventually enhouse the quantized weights
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

        # Step 1: Find 1% of the salient weight channels according to importance (hint: use torch.topk())
        importance = sum(input_feats).float()
        outlier_indices = torch.topk(
            importance, k=int(importance.shape[0] * protection_ratio), dim=0
        )[1]
        assert outlier_indices.dim() == 1

        # Step 2: Restore the 1% salient weight channels to their original FP16 values
        outlier = linear.weight.data[:, outlier_indices].clone()
        awq_linear.weight.data[:, outlier_indices] = outlier

        if linear.bias is not None:
            awq_linear.bias.data = linear.bias.data

        return awq_linear

    @classmethod
    def from_linear_salient_act(
        cls,
        linear: nn.Linear,
        input_feats: torch.Tensor,
        w_n_bits: int = 4,
        a_n_bits: int = 4,
        zero_point: bool = True,
        group_size: int = 128,
        act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
        quantize_output: bool = False,
        protection_ratio: float = 0.01,
    ):

        # Step 1: Find 1% of the salient weight channels according to importance (hint: use torch.topk())
        importance = sum(input_feats).float()
        outlier_indices = torch.topk(
            importance, k=int(importance.shape[0] * protection_ratio), dim=0
        )[1]
        assert outlier_indices.dim() == 1

        # this is a linear layer that will eventually enhouse the quantized weights
        awq_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            act_quant=act_quant + "_salient",
            quantize_output=quantize_output,
            outlier_indices=outlier_indices,
        )

        awq_linear.weight.data = pseudo_quantize_tensor(
            w=linear.weight.data,
            n_bit=w_n_bits,
            zero_point=zero_point,
            q_group_size=group_size,
        )

        """   
        # Step 2: Restore the 1% salient weight channels to their original FP16 values
        outlier = linear.weight.data[:, outlier_indices].clone()
        awq_linear.weight.data[:, outlier_indices] = outlier
        """

        if linear.bias is not None:
            awq_linear.bias.data = linear.bias.data

        return awq_linear

    @classmethod
    def from_linear_salient_weight_act(
        cls,
        linear: nn.Linear,
        input_feats: torch.Tensor,
        w_n_bits: int = 4,
        a_n_bits: int = 4,
        zero_point: bool = True,
        group_size: int = 128,
        act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
        quantize_output: bool = False,
        protection_ratio: float = 0.01,
    ):

        # Step 1: Find 1% of the salient weight channels according to importance (hint: use torch.topk())
        importance = sum(input_feats).float()
        outlier_indices = torch.topk(
            importance, k=int(importance.shape[0] * protection_ratio), dim=0
        )[1]
        assert outlier_indices.dim() == 1

        # this is a linear layer that will eventually enhouse the quantized weights
        awq_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            w_n_bits=w_n_bits,
            a_n_bits=a_n_bits,
            act_quant=act_quant + "_salient",
            quantize_output=quantize_output,
            outlier_indices=outlier_indices,
        )

        awq_linear.weight.data = pseudo_quantize_tensor(
            w=linear.weight.data,
            n_bit=w_n_bits,
            zero_point=zero_point,
            q_group_size=group_size,
        )

        # Step 2: Restore the 1% salient weight channels to their original FP16 values
        outlier = linear.weight.data[:, outlier_indices].clone()
        awq_linear.weight.data[:, outlier_indices] = outlier

        if linear.bias is not None:
            awq_linear.bias.data = linear.bias.data

        return awq_linear


def quantize_opt(
    model,
    w_n_bits: int = 4,
    a_n_bits: int = 4,
    zero_point: bool = True,
    group_size: int = 128,
    act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
    quantize_bmm_input: bool = True,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
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
                w_n_bits=w_n_bits,  # new input param
                a_n_bits=a_n_bits,  # new input param
                zero_point=zero_point,  # new input param
                group_size=group_size,  # new input param
                act_quant=act_quant,  # new input param
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
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QuantizedLinear.from_linear(
                m.k_proj,
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QuantizedLinear.from_linear(
                m.v_proj,
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = QuantizedLinear.from_linear(
                m.out_proj,
                w_n_bits=w_n_bits,  # new input param
                a_n_bits=a_n_bits,  # new input param
                zero_point=zero_point,  # new input param
                group_size=group_size,  # new input param
                act_quant=act_quant,  # new input param
            )

    return model


def quantize_opt_salient_weight_fp16(
    model,
    input_feats,
    w_n_bits: int = 4,
    a_n_bits: int = 4,
    zero_point: bool = True,
    group_size: int = 128,
    act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
    quantize_bmm_input: bool = True,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = QuantizedLinear.from_linear_salient_weight(
                m.fc1,
                input_feats["model." + name + ".fc1"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
            )
            m.fc2 = QuantizedLinear.from_linear_salient_weight(
                m.fc2,
                input_feats["model." + name + ".fc2"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QuantizedLinear.from_linear_salient_weight(
                m.q_proj,
                input_feats["model." + name + ".q_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QuantizedLinear.from_linear_salient_weight(
                m.k_proj,
                input_feats["model." + name + ".k_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QuantizedLinear.from_linear_salient_weight(
                m.v_proj,
                input_feats["model." + name + ".v_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = QuantizedLinear.from_linear_salient_weight(
                m.out_proj,
                input_feats["model." + name + ".out_proj"],
                w_n_bits=w_n_bits,  # new input param
                a_n_bits=a_n_bits,  # new input param
                zero_point=zero_point,  # new input param
                group_size=group_size,  # new input param
                act_quant=act_quant,  # new input param
            )

    return model


def quantize_opt_salient_act_fp16(
    model,
    input_feats,
    w_n_bits: int = 4,
    a_n_bits: int = 4,
    zero_point: bool = True,
    group_size: int = 128,
    act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
    quantize_bmm_input: bool = True,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = QuantizedLinear.from_linear_salient_act(
                m.fc1,
                input_feats["model." + name + ".fc1"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
            )
            m.fc2 = QuantizedLinear.from_linear_salient_act(
                m.fc2,
                input_feats["model." + name + ".fc2"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QuantizedLinear.from_linear_salient_act(
                m.q_proj,
                input_feats["model." + name + ".q_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QuantizedLinear.from_linear_salient_act(
                m.k_proj,
                input_feats["model." + name + ".k_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QuantizedLinear.from_linear_salient_act(
                m.v_proj,
                input_feats["model." + name + ".v_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = QuantizedLinear.from_linear_salient_act(
                m.out_proj,
                input_feats["model." + name + ".out_proj"],
                w_n_bits=w_n_bits,  # new input param
                a_n_bits=a_n_bits,  # new input param
                zero_point=zero_point,  # new input param
                group_size=group_size,  # new input param
                act_quant=act_quant,  # new input param
            )

    return model


def quantize_opt_salient_weight_act_fp16(
    model,
    input_feats,
    w_n_bits: int = 4,
    a_n_bits: int = 4,
    zero_point: bool = True,
    group_size: int = 128,
    act_quant: Literal["per_token", "per_tensor", "none"] = "per_token",
    quantize_bmm_input: bool = True,
    protection_ratio: float = 0.01,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = QuantizedLinear.from_linear_salient_weight_act(
                m.fc1,
                input_feats["model." + name + ".fc1"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                protection_ratio=protection_ratio,
            )
            m.fc2 = QuantizedLinear.from_linear_salient_weight_act(
                m.fc2,
                input_feats["model." + name + ".fc2"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                protection_ratio=protection_ratio,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QuantizedLinear.from_linear_salient_weight_act(
                m.q_proj,
                input_feats["model." + name + ".q_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                protection_ratio=protection_ratio,
            )
            m.k_proj = QuantizedLinear.from_linear_salient_weight_act(
                m.k_proj,
                input_feats["model." + name + ".k_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                protection_ratio=protection_ratio,
            )
            m.v_proj = QuantizedLinear.from_linear_salient_weight_act(
                m.v_proj,
                input_feats["model." + name + ".v_proj"],
                w_n_bits=w_n_bits,
                a_n_bits=a_n_bits,
                zero_point=zero_point,
                group_size=group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                protection_ratio=protection_ratio,
            )
            m.out_proj = QuantizedLinear.from_linear_salient_weight_act(
                m.out_proj,
                input_feats["model." + name + ".out_proj"],
                w_n_bits=w_n_bits,  # new input param
                a_n_bits=a_n_bits,  # new input param
                zero_point=zero_point,  # new input param
                group_size=group_size,  # new input param
                act_quant=act_quant,  # new input param
                protection_ratio=protection_ratio,
            )

    return model
