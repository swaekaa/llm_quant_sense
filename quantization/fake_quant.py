import torch


def fake_4bit_quantize(param):
    # Simple symmetric quantization simulation

    with torch.no_grad():
        max_val = param.abs().max()
        scale = max_val / 7  # 4-bit signed → [-8,7]

        q = torch.round(param / scale)
        q = torch.clamp(q, -8, 7)

        return q * scale