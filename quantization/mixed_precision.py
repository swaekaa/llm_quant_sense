import copy
import torch
import torch.nn as nn
from quantization.layerwise import fake_quant_linear_layer


def apply_mixed_precision(model, layer_bit_map):
    """
    Apply mixed-precision fake quantization based on a layer -> bit-width map.

    layer_bit_map example:
    {
        0: 6,
        1: 6,
        2: 8,
        3: 8,
        4: 8,
        5: 4
    }
    """
    model_q = copy.deepcopy(model)
    transformer = model_q.distilbert.transformer.layer

    for layer_idx, num_bits in layer_bit_map.items():
        for name, module in transformer[layer_idx].named_modules():
            if isinstance(module, nn.Linear):
                parent = transformer[layer_idx]
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                setattr(
                    parent,
                    parts[-1],
                    fake_quant_linear_layer(module, num_bits)
                )

    return model_q
