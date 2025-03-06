import torch
import math
from typing import Tuple


def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack(
            [
                sh_0_0,
            ],
            dim=-1,
        )

    sh_1_0 = math.sqrt(3) * x
    sh_1_1 = math.sqrt(3) * y
    sh_1_2 = math.sqrt(3) * z
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

    sh_2_0 = math.sqrt(15) * x * z
    sh_2_1 = math.sqrt(15) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = math.sqrt(5) * (y2 - (1 / 2) * x2z2)
    sh_2_3 = math.sqrt(15) * y * z
    sh_2_4 = (1 / 2) * math.sqrt(15) * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)

    sh_3_0 = (1 / 6) * math.sqrt(42) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(7) * sh_2_0 * y
    sh_3_2 = (1 / 8) * math.sqrt(168) * (4.0 * y2 - x2z2) * x
    sh_3_3 = (1 / 2) * math.sqrt(7) * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = (1 / 8) * math.sqrt(168) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(7) * sh_2_4 * y
    sh_3_6 = (1 / 6) * math.sqrt(42) * (sh_2_4 * z - sh_2_0 * x)

    if lmax == 3:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
            ],
            dim=-1,
        )

    sh_4_0 = (3 / 4) * math.sqrt(2) * (sh_3_0 * z + sh_3_6 * x)
    sh_4_1 = (3 / 4) * sh_3_0 * y + (3 / 8) * math.sqrt(6) * sh_3_1 * z + (3 / 8) * math.sqrt(6) * sh_3_5 * x
    sh_4_2 = (
        -3 / 56 * math.sqrt(14) * sh_3_0 * z
        + (3 / 14) * math.sqrt(21) * sh_3_1 * y
        + (3 / 56) * math.sqrt(210) * sh_3_2 * z
        + (3 / 56) * math.sqrt(210) * sh_3_4 * x
        + (3 / 56) * math.sqrt(14) * sh_3_6 * x
    )
    sh_4_3 = (
        -3 / 56 * math.sqrt(42) * sh_3_1 * z
        + (3 / 28) * math.sqrt(105) * sh_3_2 * y
        + (3 / 28) * math.sqrt(70) * sh_3_3 * x
        + (3 / 56) * math.sqrt(42) * sh_3_5 * x
    )
    sh_4_4 = -3 / 28 * math.sqrt(42) * sh_3_2 * x + (3 / 7) * math.sqrt(7) * sh_3_3 * y - 3 / 28 * math.sqrt(42) * sh_3_4 * z
    sh_4_5 = (
        -3 / 56 * math.sqrt(42) * sh_3_1 * x
        + (3 / 28) * math.sqrt(70) * sh_3_3 * z
        + (3 / 28) * math.sqrt(105) * sh_3_4 * y
        - 3 / 56 * math.sqrt(42) * sh_3_5 * z
    )
    sh_4_6 = (
        -3 / 56 * math.sqrt(14) * sh_3_0 * x
        - 3 / 56 * math.sqrt(210) * sh_3_2 * x
        + (3 / 56) * math.sqrt(210) * sh_3_4 * z
        + (3 / 14) * math.sqrt(21) * sh_3_5 * y
        - 3 / 56 * math.sqrt(14) * sh_3_6 * z
    )
    sh_4_7 = -3 / 8 * math.sqrt(6) * sh_3_1 * x + (3 / 8) * math.sqrt(6) * sh_3_5 * z + (3 / 4) * sh_3_6 * y
    sh_4_8 = (3 / 4) * math.sqrt(2) * (-sh_3_0 * x + sh_3_6 * z)
    if lmax == 4:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
            ],
            dim=-1,
        )

    sh_5_0 = (1 / 10) * math.sqrt(110) * (sh_4_0 * z + sh_4_8 * x)
    sh_5_1 = (1 / 5) * math.sqrt(11) * sh_4_0 * y + (1 / 5) * math.sqrt(22) * sh_4_1 * z + (1 / 5) * math.sqrt(22) * sh_4_7 * x
    sh_5_2 = (
        -1 / 30 * math.sqrt(22) * sh_4_0 * z
        + (4 / 15) * math.sqrt(11) * sh_4_1 * y
        + (1 / 15) * math.sqrt(154) * sh_4_2 * z
        + (1 / 15) * math.sqrt(154) * sh_4_6 * x
        + (1 / 30) * math.sqrt(22) * sh_4_8 * x
    )
    sh_5_3 = (
        -1 / 30 * math.sqrt(66) * sh_4_1 * z
        + (1 / 15) * math.sqrt(231) * sh_4_2 * y
        + (1 / 30) * math.sqrt(462) * sh_4_3 * z
        + (1 / 30) * math.sqrt(462) * sh_4_5 * x
        + (1 / 30) * math.sqrt(66) * sh_4_7 * x
    )
    sh_5_4 = (
        -1 / 15 * math.sqrt(33) * sh_4_2 * z
        + (2 / 15) * math.sqrt(66) * sh_4_3 * y
        + (1 / 15) * math.sqrt(165) * sh_4_4 * x
        + (1 / 15) * math.sqrt(33) * sh_4_6 * x
    )
    sh_5_5 = (
        -1 / 15 * math.sqrt(110) * sh_4_3 * x + (1 / 3) * math.sqrt(11) * sh_4_4 * y - 1 / 15 * math.sqrt(110) * sh_4_5 * z
    )
    sh_5_6 = (
        -1 / 15 * math.sqrt(33) * sh_4_2 * x
        + (1 / 15) * math.sqrt(165) * sh_4_4 * z
        + (2 / 15) * math.sqrt(66) * sh_4_5 * y
        - 1 / 15 * math.sqrt(33) * sh_4_6 * z
    )
    sh_5_7 = (
        -1 / 30 * math.sqrt(66) * sh_4_1 * x
        - 1 / 30 * math.sqrt(462) * sh_4_3 * x
        + (1 / 30) * math.sqrt(462) * sh_4_5 * z
        + (1 / 15) * math.sqrt(231) * sh_4_6 * y
        - 1 / 30 * math.sqrt(66) * sh_4_7 * z
    )
    sh_5_8 = (
        -1 / 30 * math.sqrt(22) * sh_4_0 * x
        - 1 / 15 * math.sqrt(154) * sh_4_2 * x
        + (1 / 15) * math.sqrt(154) * sh_4_6 * z
        + (4 / 15) * math.sqrt(11) * sh_4_7 * y
        - 1 / 30 * math.sqrt(22) * sh_4_8 * z
    )
    sh_5_9 = -1 / 5 * math.sqrt(22) * sh_4_1 * x + (1 / 5) * math.sqrt(22) * sh_4_7 * z + (1 / 5) * math.sqrt(11) * sh_4_8 * y
    sh_5_10 = (1 / 10) * math.sqrt(110) * (-sh_4_0 * x + sh_4_8 * z)
    if lmax == 5:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
            ],
            dim=-1,
        )
        
if __name__ == "__main__":
    test_sizes = [2, 3, 4, 5]
    lmax_values = [0, 1, 2, 3, 4, 5]
    
    for size in test_sizes:
        x, y, z = torch.randn(size), torch.randn(size), torch.randn(size)
        print(f"\nInput size: {size}")
        for lmax in lmax_values:
            sh_values = _spherical_harmonics(lmax, x, y, z)
            print(f"lmax={lmax}, Output shape: {sh_values.shape}")      