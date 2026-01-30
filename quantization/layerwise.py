import copy
import torch
import torch.nn as nn


def fake_quant_tensor(x, num_bits=8):
    """
    Fake quantize a tensor to simulate low-bit precision.
    """
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    scale = x.abs().max() / qmax
    if scale == 0:
        return x

    x_int = torch.clamp((x / scale).round(), qmin, qmax)
    x_quant = x_int * scale

    return x_quant


def fake_quant_linear_layer(layer, num_bits=8):
    """
    Fake quantize weights of a Linear layer.
    """
    layer = copy.deepcopy(layer)

    with torch.no_grad():
        layer.weight.data = fake_quant_tensor(layer.weight.data, num_bits)
        if layer.bias is not None:
            layer.bias.data = fake_quant_tensor(layer.bias.data, num_bits)

    return layer


def quantize_single_transformer_layer(model, layer_idx, num_bits=8):
    """
    Fake quantize only ONE transformer layer.
    """
    model_q = copy.deepcopy(model)

    transformer = model_q.distilbert.transformer.layer

    for name, module in transformer[layer_idx].named_modules():
        if isinstance(module, nn.Linear):
            parent = transformer[layer_idx]
            for attr in name.split(".")[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split(".")[-1], fake_quant_linear_layer(module, num_bits))

    return model_q
