import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Attention Weights and Gate Values")
    parser.add_argument("--attention_weights", required=True, help="Path to attention weights file (numpy format)")
    parser.add_argument("--gate_values", required=True, help="Path to gate fusion values file (numpy format)")
    parser.add_argument("--output_dir", default=".", help="Output directory for plots")
    return parser.parse_args()


def plot_matrix(matrix, title, output_file):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="viridis", aspect="auto")  # Replacing seaborn with imshow
    plt.colorbar(label="Weight")  # Adding a color scale
    plt.title(title)
    plt.xlabel("Node index")
    plt.ylabel("Node index")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    args = parse_args()

    attention_weights = np.load(args.attention_weights)
    gate_values = np.load(args.gate_values)

    plot_matrix(attention_weights, "Attention Weights", f"{args.output_dir}/attention_weights.png")
    plot_matrix(gate_values, "Gate Fusion Values", f"{args.output_dir}/gate_values.png")


if __name__ == "__main__":
    main()
