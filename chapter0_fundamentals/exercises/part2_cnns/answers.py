#%%
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_cnns', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
if MAIN:
    display_array_as_img(arr[0])
# %%
print(arr)
# %%
arr1 = einops.rearrange(arr, "b c h w -> c h (b w)")
if MAIN:
    display_array_as_img(arr1)
# %%
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
if MAIN:
    display_array_as_img(arr2)
# %%
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
if MAIN:
    display_array_as_img(arr3)
# %%
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
if MAIN:
    display_array_as_img(arr4)
# %%
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
if MAIN:
    display_array_as_img(arr5)
# %%
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
if MAIN:
    display_array_as_img(arr6)
# %%
arr7 = einops.reduce(arr.astype(float), "b c h w -> h (b w)", "max").astype(int)
if MAIN:
    display_array_as_img(arr7)
# %%
arr8 = einops.reduce(arr.astype(float), "b c h w -> h w", "min").astype(int)
if MAIN:
    display_array_as_img(arr8)
# %%
arr9 = einops.rearrange(arr[1], "c h w -> c w h")
if MAIN:
    display_array_as_img(arr9)
# %%
arr10 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
if MAIN:
    display_array_as_img(arr10)
# %%
def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, "i i->")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, "i j, j -> i")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, "i, i->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, "i, j->i j")


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)
# %%
if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )
# %%
import torch as t
from collections import namedtuple


if MAIN:
    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2, 2),
            stride=(5, 2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5,),
            stride=(1,),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4,),
            stride=(5,),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2, 3),
            stride=(5, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10,1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11,0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4,),
            stride=(6,),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")
# %%
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    stride = mat.stride()
    assert len(stride) == 2, "Input must be 2D"
    assert mat.size(0) == mat.size(1), "Input must be square"

    diag = mat.as_strided(size=(mat.size(0),), stride=(stride[0] + stride[1],))

    return diag.sum()


if MAIN:
    tests.test_trace(as_strided_trace)
# %%
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    stride = mat.stride()
    assert len(stride) == 2, "Input must be 2D"
    assert mat.size(1) == vec.size(0), "Inner dimensions must match"

    vec_1 = vec.as_strided(mat.shape, stride=(0,vec.stride(0)))

    return (mat * vec_1).sum(dim=1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)
# %%
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    sA0, sA1 = matA.stride()
    dA0, dA1 = matA.shape
    sB0, sB1 = matB.stride()
    dB0, dB1 = matB.shape

    expanded_size = (dA0, dA1, dB1)

    matA_1_stride = (sA0, sA1, 0)
    matA_1 = matA.as_strided(size=expanded_size, stride=matA_1_stride)

    matB_1_stride = (0, sB0, sB1)
    matB_1 = matB.as_strided(size=expanded_size, stride=matB_1_stride)

    return (matA_1 * matB_1).sum(dim=1)


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)
# %%
