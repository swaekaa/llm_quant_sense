import matplotlib.pyplot as plt


def plot_layer_sensitivity(sensitivity_dict, save_path=None):
    layers = list(sensitivity_dict.keys())
    drops = [v * 100 for v in sensitivity_dict.values()]

    plt.figure(figsize=(8, 4))
    plt.bar(layers, drops)
    plt.xlabel("Transformer Layer")
    plt.ylabel("Accuracy Drop (%)")
    plt.title("Layer-wise Quantization Sensitivity")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
