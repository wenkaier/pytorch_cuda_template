#include <cuda_runtime.h>
#include <cooperative_groups.h>
// #include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void cuda_add_kernel(const float *a, const float *b, float *c, const int n)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= n)
        return;
    c[idx] = a[idx] + b[idx];
}
void cuda_add(const float *a, const float *b, float *c, const int n)
{
    cuda_add_kernel<<<(n + 128 - 1) / 128, 128>>>(a, b, c, n);
}
