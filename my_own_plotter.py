#!/usr/bin/env python3
import argparse
import numpy as np
import sys

# Stretched Gaussian parameters
CUTOFF = 6.25  # cutoff from PLUMED
STRETCH_A = 1.0 / (1.0 - np.exp(-CUTOFF))
STRETCH_B = -np.exp(-CUTOFF) / (1.0 - np.exp(-CUTOFF))

def read_hills_file(file_path):
    """Parse the HILLS file and return the relevant data as arrays."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if line.startswith("#!") or "time" in line:
            continue
        data.append([float(x) for x in line.split()])

    return np.array(data)

def stretched_gaussian_1D(x, mean, sigma):
    gaussian = np.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))
    return np.maximum(gaussian, 0)

def stretched_gaussian_2D(x, y, mean_x, mean_y, sigma_x, sigma_y):
    gaussian = np.exp(
        -0.5 * ((x - mean_x) ** 2) / (sigma_x ** 2)
        -0.5 * ((y - mean_y) ** 2) / (sigma_y ** 2)
    )
    stretched = STRETCH_A * gaussian + STRETCH_B
    return np.maximum(stretched, 0)

def compute_bias_1D(hills_data, bins, min_val, max_val):
    x_vals = np.linspace(min_val, max_val, bins+1)
    bias = np.zeros_like(x_vals)
    for _, mean, sigma, height, *rest in hills_data:
        kernel = stretched_gaussian_1D(x_vals, mean, sigma)
        bias += height * kernel
    return x_vals, -bias

def compute_bias_2D(hills_data, bins_x, bins_y, min_x, max_x, min_y, max_y):
    x_vals = np.linspace(min_x, max_x, bins_x+1)
    y_vals = np.linspace(min_y, max_y, bins_y+1)
    bias = np.zeros((bins_x+1, bins_y+1))
    for row in hills_data:
        _, mean_x, mean_y, sigma_x, sigma_y, height, *rest = row
        for i, x in enumerate(x_vals):
            kernel = stretched_gaussian_2D(x, y_vals, mean_x, mean_y, sigma_x, sigma_y)
            bias[i, :] += height * kernel
    return x_vals, y_vals, -bias

def parse_list_of(type_, vals, name):
    try:
        return [type_(v) for v in vals]
    except ValueError:
        print(f"Error: all values for {name} must be of type {type_.__name__}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Compute 1D or 2D bias from a HILLS file."
    )
    parser.add_argument(
        "-d", "--dim",
        type=int, choices=[1, 2], required=True,
        help="Dimension: 1 or 2"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input HILLS file"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to output file (default: '<input>_bias.txt')"
    )
    parser.add_argument(
        "-bins", "--bins",
        nargs="+", type=int, default=[300, 300],
        help="Number of bins: one integer for 1D; two integers for 2D"
    )
    parser.add_argument(
        "-min", "--min",
        nargs="+", required=True,
        help="Minimum value(s): one float for 1D; two floats for 2D"
    )
    parser.add_argument(
        "-max", "--max",
        nargs="+", required=True,
        help="Maximum value(s): one float for 1D; two floats for 2D"
    )
    args = parser.parse_args()

    # derive output filename
    out_file = args.output or args.input.rsplit(".",1)[0] + "_bias.txt"
    hills_data = read_hills_file(args.input)

    # parse bins and ranges
    if args.dim == 1:
        if len(args.bins) != 1 or len(args.min) != 1 or len(args.max) != 1:
            parser.error("For 1D, specify exactly one value for -bins, -min, and -max")
        bins = args.bins[0]
        min_val = float(args.min[0])
        max_val = float(args.max[0])

        x, bias = compute_bias_1D(hills_data, bins, min_val, max_val)
        with open(out_file, "w") as f:
            for xi, bi in zip(x, bias):
                f.write(f"{xi:.8f} {bi:.8f}\n")

    else:  # 2D
        if len(args.bins) != 2 or len(args.min) != 2 or len(args.max) != 2:
            parser.error("For 2D, specify exactly two values for -bins, -min, and -max")
        bins_x, bins_y = args.bins
        min_x, min_y = parse_list_of(float, args.min, "-min")
        max_x, max_y = parse_list_of(float, args.max, "-max")

        x, y, bias = compute_bias_2D(
            hills_data, bins_x, bins_y,
            min_x, max_x, min_y, max_y
        )
        with open(out_file, "w") as f:
            for j, yj in enumerate(y):
                for i, xi in enumerate(x):
                    f.write(f"{xi:.8f} {yj:.8f} {bias[i,j]:.8f}\n")

    print(f"{args.dim}D bias written to {out_file}")
    
if __name__ == "__main__":
    main()
