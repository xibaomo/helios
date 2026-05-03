#include <cuComplex.h>

#include "xmux.h"

template <typename T, typename Func, typename... Args>
__global__ void ops_each_knl(Func f, T* d_A, int n, Args... others) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    d_A[i] = f(d_A[i], others[i]...);
  }
}

template <typename T>
__device__ __forceinline__ T add_elem(T a, T b) {
  if constexpr (std::is_same_v<T, cuComplex>) {
    return cuCaddf(a, b);
  } else {
    return a + b;
  }
}

template <typename Arr>
void XMux<Arr>::add(const XMux<Arr>& other) {
  auto op = [] __device__(cuComplex & a, cuComplex & b) { return add_elem(a, b); };
  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (cuComplex*)m_device_data,
                                          (int)m_size,
                                          (cuComplex*)other.device_data());
  cudaDeviceSynchronize();
}