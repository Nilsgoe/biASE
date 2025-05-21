import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cmocean.cm as cmo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot 1D/2D bias with optional periodic wrapping."
    )
    parser.add_argument(
        "-d", "--dim", type=int, choices=[1, 2], required=True,
        help="Dimension: 1 for 1D or 2 for 2D plot"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input bias file (e.g., <input>_bias.txt)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output figure file (e.g., FES.png). Defaults to <input>_fes.png"
    )
    parser.add_argument(
        "-bins", nargs='+', type=int, default=[300, 300],
        help="Number of bins: 1 int for 1D, 2 ints for 2D"
    )
    return parser.parse_args()


def load_and_reshape(input_file, dim, bins):
    data = np.loadtxt(input_file)
    if dim == 1:
        # Expect two columns: coordinate, fes
        coords = data[:, 0]
        fes = data[:, 1]
        return coords, fes
    else:
        # Expect three columns: x, y, fes; reshape into grid
        x_vals = data[:, 0].reshape(bins[0]+1, bins[1]+1)
        y_vals = data[:, 1].reshape(bins[0]+1, bins[1]+1)
        fes = data[:, 2].reshape(bins[0]+1, bins[1]+1)
        return x_vals, y_vals, fes


def plot_1d(coords, fes, args):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coords, fes, label='Free Energy Surface')
    ax.set_xlabel('Collective Variable', fontsize=14)
    ax.set_ylabel('FES', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_2d(x_vals, y_vals, fes, args):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.contourf(x_vals, y_vals, fes, 15, cmap=cmo.tempo)
    cp = ax.contour(x_vals, y_vals, fes, 15,
                    linestyles='--', colors='grey', linewidths=1.3)
    ax.set_xlabel('CV1', fontsize=14)
    ax.set_ylabel('CV2', fontsize=14)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('FES', fontsize=12)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    # Determine output filename
    if args.output:
        out_fig = args.output
    else:
        base = args.input.rsplit('.', 1)[0]
        out_fig = f"{base}_fes.png"

    # Load data
    if args.dim == 1:
        coords, fes = load_and_reshape(args.input, 1, args.bins)
        fig = plot_1d(coords, fes, args)
    else:
        x_vals, y_vals, fes = load_and_reshape(args.input, 2, args.bins)
        fig = plot_2d(x_vals, y_vals, fes, args)

    # Save figure
    fig.savefig(out_fig, dpi=300)
    print(f"Saved FES plot to {out_fig}")

if __name__ == '__main__':
    main()