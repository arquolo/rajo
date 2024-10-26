__all__ = ['laplace_kernel', 'pascal_triangle', 'rgb2gray_kernel']

import cv2
import numpy as np
import torch
from glow import pascal
from torch import Tensor


def laplace_kernel(size: int, /, *, normalize: bool = True) -> Tensor:
    """
    Calculate kernel weights for OpenCV's 2D Laplacian.
    `size` should be odd and in [1 ... 31] range.
    """
    if size % 2 != 1:
        raise ValueError(f'Kernel size should be odd. Got {size}')
    if size > 31:
        raise ValueError(
            'Kernel size should be less or equal then 31. ' f'Got {size}'
        )

    effective_size = max(3, size)
    offset = effective_size // 2
    im = np.zeros((2 * effective_size - 1, 2 * effective_size - 1), 'f4')
    im[effective_size - 1, effective_size - 1] = 1

    # Retrieve kernel from cv2
    # TODO: use pure torch
    scale = size / (4**size) if normalize else 1
    im = cv2.Laplacian(im, cv2.CV_32F, ksize=size, scale=scale)

    return torch.as_tensor(
        im[offset : offset + effective_size, offset : offset + effective_size]
    )


def rgb2gray_kernel() -> Tensor:
    """
    Calculate 1x3 kernel for OpenCV's RGB to Gray transform.
    """
    w = np.eye(3, dtype='f4')[None, :, :]  # (1 3 3)
    w = cv2.cvtColor(w, cv2.COLOR_RGB2GRAY)  # (1 3)
    return torch.from_numpy(w)


def pascal_triangle(n: int) -> Tensor:
    return torch.from_numpy(pascal(n))
