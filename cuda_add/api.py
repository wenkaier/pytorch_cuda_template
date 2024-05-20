import os
import os.path as osp
import torch
from torch.utils.cpp_extension import load
from torch import FloatTensor

curr_path = os.path.abspath(os.path.dirname(__file__))

build_directory = osp.join(curr_path, "build")
os.makedirs(build_directory, exist_ok=True)

cuda_module = load(
    name="cuda_add",
    sources=[
        osp.join(curr_path, "cuda_add.cpp"),
        osp.join(curr_path, "cuda_add.cu"),
    ],
    build_directory=build_directory,
    verbose=True,
)


def cuda_add(a: FloatTensor, b: FloatTensor):
    c = torch.zeros_like(a)
    cuda_module.cuda_add(a, b, c, len(a))
    return c
