#include "xmux.h"

// template <typename T, typename Func, typename... Args>
// __global__ void ops_each_knl(T* d_A, int n, Func f, Args... others) {
//   int idx = blockDim.x * blockIdx.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = idx; i < n; i += stride) {
//     d_A[i] = f(d_A[i], others[i]...);
//   }
// }

// template <typename Arr, typename F, typename... OtherArrs>
// void XMux<Arr>::for_each(F fn, XMux<OtherArrs>&... others) {
//   ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(m_device_data, m_size, fn,
//                                           others.device_data()...);
//   cudaDeviceSynchronize();
// }