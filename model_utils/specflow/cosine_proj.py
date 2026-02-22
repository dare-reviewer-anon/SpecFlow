import torch
from functools import lru_cache
from typing import Tuple


def _validate_4d(x: torch.Tensor, name: str) -> Tuple[int, int, int, int]:
    if x.dim() != 4:
        raise ValueError(f"{name} must be a 4D tensor (B,C,H,W). Got shape={tuple(x.shape)}")
    return x.shape  # type: ignore[return-value]


@lru_cache(maxsize=64)
def _dct_basis_cpu(N: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Orthonormal DCT-II basis on CPU. Cached by (N, dtype).
    Returned tensor is on CPU; caller moves to device as needed.
    """
    # Build with float64 for accuracy, then cast.
    n = torch.arange(N, dtype=torch.float64).reshape(1, N)
    k = torch.arange(N, dtype=torch.float64).reshape(N, 1)
    mat = torch.cos(torch.pi / N * (n + 0.5) * k)

    # Orthonormal scaling
    mat[0, :] *= 1.0 / (N ** 0.5)
    mat[1:, :] *= (2.0 / N) ** 0.5

    return mat.to(dtype=dtype)


def dct_matrix(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix D (N x N), on (device, dtype).
    """
    return _dct_basis_cpu(N, dtype).to(device=device)


def idct_matrix(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Inverse of orthonormal DCT-II is transpose.
    """
    return dct_matrix(N, device, dtype).transpose(0, 1)


def block_dct(x: torch.Tensor, block_size: int) -> torch.Tensor:
    B, C, H, W = _validate_4d(x, "x")
    b = int(block_size)
    if b <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if H % b != 0 or W % b != 0:
        raise ValueError(f"H,W must be divisible by block_size. Got H={H}, W={W}, b={b}")

    D = dct_matrix(b, x.device, x.dtype)  # (b,b)

    # (B,C,Hb,b,Wb,b)
    x_blocks = x.view(B, C, H // b, b, W // b, b)
    # -> (B,C,Hb,Wb,b,b)
    x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()

    # 2D DCT: D * X * D^T, applied to the last two dims
    # left multiply on rows
    X = torch.matmul(D, x_blocks)  # (B,C,Hb,Wb,b,b)
    # right multiply on cols
    X = torch.matmul(X, D.transpose(0, 1))

    # Restore to (B,C,H,W)
    X = X.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return X


def block_idct(X: torch.Tensor, block_size: int) -> torch.Tensor:
    B, C, H, W = _validate_4d(X, "X")
    b = int(block_size)
    if b <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if H % b != 0 or W % b != 0:
        raise ValueError(f"H,W must be divisible by block_size. Got H={H}, W={W}, b={b}")

    # For orthonormal DCT: inverse is transpose
    ID = idct_matrix(b, X.device, X.dtype)  # (b,b) = D^T
    D = ID.transpose(0, 1)                  # (b,b) = D

    # (B,C,Hb,b,Wb,b)
    X_blocks = X.view(B, C, H // b, b, W // b, b)
    # -> (B,C,Hb,Wb,b,b)
    X_blocks = X_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()

    # 2D IDCT: D^T * X * D
    x = torch.matmul(ID, X_blocks)
    x = torch.matmul(x, D)

    # Restore to (B,C,H,W)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return x


