#!/usr/bin/env python3
import argparse
import numpy as np
import sys

# Stretched Gaussian parameters
CUTOFF    = 6.25
STRETCH_A = 1.0 / (1.0 - np.exp(-CUTOFF))
STRETCH_B = -np.exp(-CUTOFF) / (1.0 - np.exp(-CUTOFF))

def read_hills_file(path):
    with open(path) as f:
        lines = [l for l in f if not (l.startswith("#!") or "time" in l)]
    return np.array([list(map(float, l.split())) for l in lines])

def stretched_gaussian_1D(x, mean, sigma, stretchA, stretchB, bias_h, wrap, bounds):
    if wrap:
        # minimal circular diff in 1D
        low, high = bounds
        period = high - low
        diffs = (x - mean + 0.5*period) % period - 0.5*period
        vals = np.exp(-0.5*(diffs**2)/(sigma**2))
    else:
        vals = np.exp(-0.5*((x - mean)**2)/(sigma**2))
    if stretchA is not None:
        vals = np.maximum(stretchA*vals + stretchB, 0.0)
    return bias_h *vals 

def stretched_gaussian_2D(x, y, mean_x, mean_y, sigma_x, sigma_y,
                          stretchA, stretchB, bias_h, wrap, bounds):
    if wrap:
        (low_x, high_x, low_y, high_y) = bounds
        per_x = high_x - low_x
        per_y = high_y - low_y
        dx = (x - mean_x + 0.5*per_x) % per_x - 0.5*per_x
        dy = (y - mean_y + 0.5*per_y) % per_y - 0.5*per_y
    else:
        dx = x - mean_x
        dy = y - mean_y

    vals = np.exp(-0.5*(dx**2)/(sigma_x**2)
                  -0.5*(dy**2)/(sigma_y**2))
    ''' print(vals)
    print(bias_h * vals )
    exit()'''
    if stretchA is not None:
        vals = np.maximum(stretchA*vals + stretchB, 0.0)
    return bias_h * vals

def compute_bias_1D(hills, bins, mn, mx, wrap, bounds):
    x = np.linspace(mn, mx, bins+1)
    bias = np.zeros_like(x)
    for row in hills:
        _, mean, sigma, height, *rest = row
        bias += stretched_gaussian_1D(
            x, mean, sigma, STRETCH_A, STRETCH_B,
            height, wrap, bounds,
        )
    return x, -bias

def compute_bias_2D(hills, bins_x, bins_y, mnx, mxx, mny, mxy, wrap, bounds):
    x = np.linspace(mnx, mxx, bins_x+1)
    y = np.linspace(mny, mxy, bins_y+1)
    bias = np.zeros((bins_x+1, bins_y+1))
    for row in hills:
        _, mx, my, sx, sy, height, *rest = row
        for i, xv in enumerate(x):
            bias[i,:] += stretched_gaussian_2D(
                xv, y, mx, my, sx, sy,
                STRETCH_A, STRETCH_B, height,
                wrap, bounds,
            )
    return x, y, -bias

def main():
    parser = argparse.ArgumentParser(
        description="Compute 1D/2D bias with optional periodic wrapping."
    )
    parser.add_argument("-d","--dim", type=int, choices=[1,2],
                        required=True, help="1D or 2D")
    parser.add_argument("-i","--input", required=True,
                        help="HILLS input file")
    parser.add_argument("-o","--output", default=None,
                        help="Output file (default: <input>_bias.txt)")
    parser.add_argument("-bins", nargs="+", type=int, default=[300,300],
                        help="Bins: 1 int for 1D, 2 ints for 2D")
    parser.add_argument("-min", nargs="+", type=float, required=True,
                        help="Min: 1 float for 1D, 2 for 2D")
    parser.add_argument("-max", nargs="+", type=float, required=True,
                        help="Max: 1 float for 1D, 2 for 2D")
    parser.add_argument("-w","--wrapped", type=bool, default=False,
                        choices=[True,False],
                        help="Enable periodic wrapping")
    parser.add_argument("--bounds", nargs="+", type=float, default=None,
                        help="Bounds for wrapping: [low,high] for 1D; [low_x,high_x,low_y,high_y] for 2D")

    args = parser.parse_args()

    # Validate bounds if wrapping
    if args.wrapped:
        need = 2 if args.dim==1 else 4
        if not args.bounds or len(args.bounds)!=need:
            parser.error(f"--bounds must have {need} values when --wrapped is set")

    out = args.output or args.input.rsplit(".",1)[0] + "_bias.txt"
    hills = read_hills_file(args.input)

    if args.dim==1:
        if len(args.bins)!=1 or len(args.min)!=1 or len(args.max)!=1:
            parser.error("1D requires exactly one value for -bins, -min, -max")
        x, bias = compute_bias_1D(
            hills,
            args.bins[0],
            args.min[0], args.max[0],
            args.wrapped,
            tuple(args.bounds) if args.wrapped else None
        )
        with open(out,"w") as f:
            for xv,bv in zip(x,bias):
                f.write(f"{xv:.8f} {bv:.8f}\n")

    else:
        if len(args.bins)!=2 or len(args.min)!=2 or len(args.max)!=2:
            parser.error("2D requires exactly two values for -bins, -min, -max")
        x,y,bias = compute_bias_2D(
            hills,
            args.bins[0], args.bins[1],
            args.min[0], args.max[0],
            args.min[1], args.max[1],
            args.wrapped,
            tuple(args.bounds) if args.wrapped else None
        )
        with open(out,"w") as f:
            for j,yv in enumerate(y):
                for i,xv in enumerate(x):
                    f.write(f"{xv:.8f} {yv:.8f} {bias[i,j]:.8f}\n")

    print(f"{args.dim}D bias written to {out}")

if __name__=="__main__":
    main()
