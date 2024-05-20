#include <torch/extension.h>
#include "cuda_add.h"

void torch_launch_cuda_add(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c, const int n)
{
    cuda_add((const float *)a.data_ptr(), (const float *)b.data_ptr(), (float *)c.data_ptr(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_add", &torch_launch_cuda_add, "cuda_add kernel warpper");
}
