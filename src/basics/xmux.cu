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
template <typename T, typename U>
__global__ void set_diagonal_grid_stride_kernel(T* __restrict__ matrix, U value,
                                                int cols) {
  // 1. Calculate the unique global thread ID
  int start_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. Calculate the total number of threads in the entire grid (the stride)
  int stride = blockDim.x * gridDim.x;

  // 3. Loop across the data jumping by 'stride' each iteration
  for (int idx = start_idx; idx < cols; idx += stride) {
    int diagonal_offset = idx * cols + idx;
    if constexpr (std::is_same_v<T, CUDA_COMPLEX> && std::is_same_v<U, float>)
      matrix[diagonal_offset] = make_cuda_complex(value, 0.f);
    if constexpr (std::is_same_v<T, float> && std::is_same_v<U, float>)
      matrix[diagonal_offset] = value;
    if constexpr (std::is_same_v<T, CUDA_COMPLEX> && std::is_same_v<U, Complex>)
      matrix[diagonal_offset] = make_cuda_complex(value.real(), value.imag());
  }
}

template <typename Arr>
void XMux<Arr>::eye() {
  zero();
  if constexpr (XMux::is_2D::value) {
    assert(this->m_size1 == this->m_size2);
    if (m_dev == Device::__cpu__) {
      m_cpu->eye();
    }
    if (m_dev == Device::__gpu__) {
      set_diagonal_grid_stride_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
          (dev_dtype*)this->m_device_data, (float)1.f, this->m_size2);
    }
    // ??????????
    cudaDeviceSynchronize();

    // ???????????????(????????)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
  }
}

template <typename Arr>
void XMux<Arr>::add(const XMux<Arr>& other) {
  auto op = [] __device__(auto& a, auto& b) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, CUDA_COMPLEX>) {
      return cu_add(a, b);
    } else {
      return a + b;
    }
  };
  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (CUDA_COMPLEX*)m_device_data,
                                          (int)m_size,
                                          (CUDA_COMPLEX*)other.device_data());
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::substract(const XMux<Arr>& other) {
  auto op = [] __device__(auto& a, auto& b) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, CUDA_COMPLEX>) {
      return cu_sub(a, b);
    } else {
      return a - b;
    }
  };
  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (CUDA_COMPLEX*)m_device_data,
                                          (int)m_size,
                                          (CUDA_COMPLEX*)other.device_data());
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::scale(Real s) {
  if (m_dev == Device::__cpu__) {
    std::cerr << "scale on cpu not supported. Real" << std::endl;
  }
  // using T = typename Arr::dtype;

  auto op = [s] __device__(auto& a) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, CUDA_COMPLEX>) {
      CUDA_COMPLEX f = make_cuda_complex(cu_real(a) * s, cu_imag(a) * s);
      return f;
    } else {
      // float/double
      return a * s;
    }
  };

  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (CUDA_COMPLEX*)m_device_data,
                                          (int)m_size);
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::scale(Complex s) {
  static_assert(!std::is_same_v<typename Arr::dtype, float>,
                "Real matrix cannot be scaled by complex");
  if (m_dev == Device::__cpu__) {
    std::cerr << "scale on cpu not supported. Complex" << std::endl;
  }
  // using T = typename Arr::dtype;
  CUDA_COMPLEX g_s = make_cuda_complex(s.real(), s.imag());

  auto op = [g_s] __device__(auto& a) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_same_v<T, CUDA_COMPLEX>) {
      CUDA_COMPLEX f = cu_mul(a, g_s);
      return f;
    }
  };

  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (CUDA_COMPLEX*)m_device_data,
                                          (int)m_size);
  cudaDeviceSynchronize();
}

template <typename Arr>
void XMux<Arr>::ones() {
  auto op = [] __device__(CUDA_COMPLEX a) { return make_cuda_complex(1.f, 0.f); };

  ops_each_knl<<<GRID_SIZE, BLOCK_SIZE>>>(op, (CUDA_COMPLEX*)m_device_data,
                                          (int)m_size);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}