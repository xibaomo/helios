#include <cuComplex.h>
#include <cassert>

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
__global__ void set_diagonal_grid_stride_kernel(T* __restrict__ matrix,
                                                Real value, int cols) {
  // 1. Calculate the unique global thread ID
  int start_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. Calculate the total number of threads in the entire grid (the stride)
  int stride = blockDim.x * gridDim.x;

  // 3. Loop across the data jumping by 'stride' each iteration
  for (int idx = start_idx; idx < cols; idx += stride) {
    int diagonal_offset = idx * cols + idx;
    if constexpr (std::is_same_v<T,cuComplex>)
      matrix[diagonal_offset] = make_cuComplex(value,0.f);
    if constexpr (std::is_same_v<T, float>)
      matrix[diagonal_offset] = value;
  }
}

template <typename Arr>
void XMux<Arr>::eye() {
  if constexpr (XMux::is_2D::value) {
    assert(this->m_size1 == this->m_size2);
    if (m_dev == Device::__cpu__) {
      m_cpu->eye();
    }
    if (m_dev == Device::__gpu__) {
      set_diagonal_grid_stride_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
          (dev_dtype*)this->m_device_data, 1.f, this->m_size2);
    }
  }
}

template <typename Arr>
void XMux<Arr>::add(const XMux<Arr>& other) {
  auto op = [] __device__(auto& a, auto& b) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, cuComplex>) {
      return cuCaddf(a, b);
    } else {
      return a + b;
    }
  };
  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (cuComplex*)m_device_data,
                                          (int)m_size,
                                          (cuComplex*)other.device_data());
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::substract(const XMux<Arr>& other) {
  auto op = [] __device__(auto& a, auto& b) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, cuComplex>) {
      return cuCsubf(a, b);
    } else {
      return a - b;
    }
  };
  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (cuComplex*)m_device_data,
                                          (int)m_size,
                                          (cuComplex*)other.device_data());
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::scale(Real s) {
  if (m_dev == Device::__cpu__) {
    std::cerr << "scale on cpu not supported" << std::endl;
  }
  // using T = typename Arr::dtype;

  auto op = [s] __device__(auto& a) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, cuComplex>) {
      cuComplex f = make_cuComplex(cuCrealf(a) * s, cuCimagf(a) * s);
      return f;
    } else {
      // float/double
      return a * s;
    }
  };

  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (cuComplex*)m_device_data,
                                          (int)m_size);
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::ones() {
  auto op = [] __device__(cuComplex a) { return make_cuComplex(1.f, 0.f); };

  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (cuComplex*)m_device_data,
                                          (int)m_size);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}