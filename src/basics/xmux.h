/**
 * This file defines class XMux. XMux is desgiend to be a shell of CPU array
 * with additional device pointer. The shell should be mapped to a certain cpu
 * array, like human's skin, operations on the skin is equivalent to operating
 * on the human, but cannot exchange skins between humans.
 *
 * In most cases, XMux is bound to an existing CPU array. In minor cases, XMux
 * may hold its own array. After creation of XMux, its m_cpu must point to an
 * array (external or internal) and no longer changes.
 */
#pragma once
#include "arr.h"
#include "cuhandlemgr.h"
#include "types.h"
#define GRID_SIZE 512
#define BLOCK_SIZE 512

template <typename T>
void transpose_gpu(int m, int n, T* d_A, T* d_B) {
  if constexpr (!std::is_same_v<T, cuComplex> && !std::is_same_v<T, Complex>) {
    std::cerr << "transpose not implemented for non-Complex" << std::endl;
    return;
  }

  auto& handle = CuHandleMgr::getInstance().getCuBlasHandle();
  void* d_b = nullptr;
  if (d_B == d_A) {
    // need extra buffer
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(cuComplex) * m * n));
  } else {
    d_b = d_B;
  }

  // exmaple, convert A=[1,2,3;4,5,6] from row-major 2x3 to col-major 2x3, so
  // m=2, n=3.
  // After copy from host to device, cublas takes A as col-major, so n
  // is the leading dim of A. in other words, in the eye of cublas,
  //  A is [1,4;
  //        2,5;
  //        3,6], then transpose can give 2x3 col-major we want
  // if still take m as A's leading dim, it gives wrong result.
  CUBLAS_CHECK(cublasCgeam(handle,
                           CUBLAS_OP_T,  // transpose A
                           CUBLAS_OP_N,  // no op for b
                           m,            // 1st dim of C
                           n,            // 2nd dim of C
                           &CUCOMPLEX_ONE, (cuComplex*)d_A,
                           n,  // leading dim of A in cublas' eye
                           &CUCOMPLEX_ZERO, nullptr,
                           m,  // B is ignored since beta=0
                           (cuComplex*)d_b,
                           m  // leading dim of C
                           ));
  if (d_B == d_A) {
    CUDA_CHECK(cudaMemcpy(d_A, d_b, sizeof(cuComplex) * m * n,
                          cudaMemcpyDeviceToDevice));
    cudaFree(d_b);
  }
}

struct Empty {
  using is_2D = std::false_type;
};
struct Dim2D {
  size_t m_size1, m_size2;
  using is_2D = std::true_type;
};
template <typename T>
using OptionalDim =
    std::conditional_t<std::is_base_of_v<Array2D<typename T::dtype>, T>, Dim2D,
                       Empty>;

template <typename Arr>
class XMux : public OptionalDim<Arr> {
 public:
  enum class Device { __cpu__, __gpu__ };
  using dtype = typename Arr::dtype;

 private:
  std::unique_ptr<Arr> m_own_cpu;
  Arr* const m_cpu;  // bound to an address, but content may change
  mutable dtype* m_device_data = nullptr;  // col major on gpu
  size_t m_size;
  mutable Device m_dev = Device::__gpu__;

  static bool is_internal(const XMux& other) {
    return other.m_own_cpu && (other.m_cpu == other.m_own_cpu.get());
  }

 public:
  ~XMux() {
    if (m_device_data) {
      CUDA_CHECK(cudaFree(m_device_data));
    }
  }

  XMux(size_t s1 = 0, size_t s2 = 0)
      : m_own_cpu(std::make_unique<Arr>()),
        m_cpu(m_own_cpu.get()),
        m_dev(Device::__gpu__) {
    if constexpr (XMux::is_2D::value) {
      if (s1 > 0 && s2 == 0) s2 = 1;
      this->m_size1 = s1;
      this->m_size2 = s2;
      this->m_size = s1 * s2;
    } else {
      m_size = s1;
    }
    CUDA_CHECK(cudaMalloc(&m_device_data, sizeof(dtype)*m_size));
  }
  XMux(const Arr& arr) : m_cpu(const_cast<Arr*>(&arr)) {
    m_dev = Device::__cpu__;
    m_size = arr.getSize();
    if constexpr (XMux::is_2D::value) {
      this->m_size1 = arr.getSize1();
      this->m_size2 = arr.getSize2();
    }
  }

  XMux(const XMux& other)
      : m_own_cpu(std::make_unique<Arr>()), m_cpu(m_own_cpu.get()) {
    m_size = other.m_size;
    m_dev = other.m_dev;
    if constexpr (XMux::is_2D::value) {
      this->m_size1 = other.m_size1;
      this->m_size2 = other.m_size2;
    }
    if (other.m_dev == Device::__cpu__) {
      *m_cpu = *(other.m_cpu);
    }

    if (other.m_dev == Device::__gpu__) {
      if (other.m_device_data) {
        CUDA_CHECK(cudaMalloc(&m_device_data, sizeof(dtype) * m_size));
        CUDA_CHECK(cudaMemcpy(m_device_data, other.m_device_data,
                              sizeof(dtype) * m_size,
                              cudaMemcpyDeviceToDevice));
      }
    }
  }
  // copy assignment
  XMux& operator=(const XMux& other) {
    if (this != &other) {
      m_size = other.m_size;
      m_dev = other.m_dev;
      if constexpr (XMux::is_2D::value) {
        this->m_size1 = other.m_size1;
        this->m_size2 = other.m_size2;
      }
      if (m_device_data) {
        CUDA_CHECK(cudaFree(m_device_data));
        m_device_data = nullptr;
      }
      if (other.m_dev == Device::__cpu__) {
        if (other.m_cpu) {  // both have hosts, copy host
          *(this->m_cpu) = *(other.m_cpu);

        } else {  // both has no host, report error
          std::cerr << "Error: the other has no valid host, cannot copy!"
                    << std::endl;
        }
      }
      if (other.m_dev == Device::__gpu__) {
        CUDA_CHECK(cudaMalloc(&m_device_data, sizeof(dtype) * m_size));
        CUDA_CHECK(cudaMemcpy(&m_device_data, other.m_device_data,
                              sizeof(dtype) * m_size,
                              cudaMemcpyDeviceToDevice));
      }
    }
    return *this;
  }

  // move constructor
  XMux(XMux&& other) noexcept
      : OptionalDim<Arr>(std::move(other)),
        m_own_cpu(is_internal(other) ? std::move(other.m_own_cpu) : nullptr),
        m_cpu(is_internal(other) ? m_own_cpu.get() : other.m_cpu),
        m_device_data(other.m_device_data),
        m_dev(other.m_dev) {
    other.m_own_cpu = nullptr;
    other.m_size = 0;
    other.m_device_data = nullptr;
  }

  XMux& operator=(XMux&& other) noexcept {
    if (this != &other) {
      m_size = other.m_size;
      m_dev = other.m_dev;
      if constexpr (XMux::is_2D::value) {
        this->m_size1 = other.m_size1;
        this->m_size2 = other.m_size2;
      }

      if (m_dev == Device::__gpu__) {
        if (m_device_data) CUDA_CHECK(cudaFree(m_device_data));
        m_device_data = other.m_device_data;
      }
      if (m_dev == Device::__cpu__) {
        *m_cpu = std::move(*other.m_cpu);
      }
      other.m_device_data = nullptr;
      other.m_own_cpu.reset();
      other.m_own_cpu = nullptr;
      other.m_size = 0;
    }
    return *this;
  }
  bool isOwn() const {
    if (m_own_cpu) return true;
    return false;
  }

  //   void resize(size_t rows, size_t cols) {
  //     m_size = rows * cols;
  //     size_t old_size = m_size;
  //     if constexpr (XMux::is_2D::value) {
  //       this->m_size1 = rows;
  //       this->m_size2 = cols;
  //     }
  //     m_size = rows * cols;
  //     if (m_size == 0) {
  //       m_own_cpu.reset();
  //       return;
  //     }
  //     if (m_dev == Device::__cpu__) {
  //       if constexpr (XMux::is_2D::value) {
  //         m_cpu->resize(this->m_size1, this->m_size2);
  //       } else {
  //         m_cpu->resize(this->m_size);
  //       }
  //     }
  //     if (m_dev == Device::__gpu__) {
  //       if (old_size != m_size) {
  //         CUDA_CHECK(cudaFree(m_device_data));
  //       }
  //       CUDA_CHECK(cudaMalloc(&m_device_data, sizeof(dtype) * m_size));
  //     }
  //   }

  size_t getSize1() const {
    if constexpr (XMux::is_2D::value) {
      return this->m_size1;
    } else {
      return m_size;
    }
  }

  size_t getSize2() const {
    if constexpr (XMux::is_2D::value) {
      return this->m_size2;
    } else {
      return 0;
    }
  }

  size_t getSize() const { return m_size; }
  Device getDevice() const { return m_dev; }
  dtype* device_data() { return m_device_data; }
  void setDeviceData(dtype* d_x) { m_device_data = d_x; }
  const dtype* device_data() const { return m_device_data; }

  void to_cpu(bool is_transpose = true) {
    if (m_dev == Device::__gpu__ && m_device_data) {
      cudaStreamSynchronize(nullptr);
      if constexpr (XMux::is_2D::value) {
        if (this->m_size1 > 1 && this->m_size2 > 1 && is_transpose) {
          transpose_gpu(this->m_size2, this->m_size1, m_device_data,
                        m_device_data);
        }
      }
      m_dev = Device::__cpu__;
      if (!m_cpu || m_cpu->getSize() != m_size) {
        if constexpr (XMux::is_2D::value) {
          m_cpu->resize(this->m_size1, this->m_size2);
        } else {
          m_cpu->resize(m_size);
        }
      }
      CUDA_CHECK(cudaMemcpy(m_cpu->getData(), m_device_data,
                            sizeof(dtype) * m_size, cudaMemcpyDeviceToHost));
    }
  }
  Arr& cpu() {
    to_cpu();
    return *m_cpu;
  }
  const Arr& cpu() const {
    if (m_dev == Device::__gpu__) {
      std::cerr << "GPU data is newer" << std::endl;
    }
    return *m_cpu;
  }

  void to_gpu(bool is_transpose = true) const {
    if (m_dev == Device::__gpu__) return;
    if (!m_device_data) {
      CUDA_CHECK(cudaMalloc(&m_device_data, sizeof(dtype) * m_size));
    }
    CUDA_CHECK(cudaMemcpy(m_device_data, m_cpu->getData(),
                          sizeof(dtype) * m_size, cudaMemcpyHostToDevice));

    if constexpr (XMux::is_2D::value) {
      if (this->m_size1 > 1 && this->m_size2 > 1 && is_transpose) {
        transpose_gpu(this->m_size1, this->m_size2, m_device_data,
                      m_device_data);
      }
    }
    m_dev = Device::__gpu__;
  }

  void setZero() {
    if (m_dev == Device::__cpu__) {
      m_cpu->zero();
    }
    if (m_dev == Device::__gpu__) {
      CUDA_CHECK(cudaMemset(m_device_data, 0, sizeof(dtype) * m_size));
    }
  }

  // fill 'other' to sub block [is:is+other.s1,js:js+other.s2]
  void fillBlock(size_t is, size_t js, const XMux<Arr>& other) {
    if constexpr (XMux::is_2D::value) {
      size_t width_bytes = other.getSize1() * sizeof(dtype);
      size_t spitch = other.getSize1() * sizeof(dtype);
      size_t dpitch = this->m_size1 * sizeof(dtype);

      // starting address in A
      dtype* d_A_offset = m_device_data + (js * this->m_size1 + is);

      const dtype* pb = (const dtype*)other.device_data();
      dtype* d_b = const_cast<dtype*>(pb);

      CUDA_CHECK(cudaMemcpy2D(d_A_offset, dpitch, (dtype*)d_b, spitch,
                              width_bytes, other.getSize2(),
                              cudaMemcpyDeviceToDevice));
      cudaDeviceSynchronize();
    }
  }

  void touchGPU() { m_dev = Device::__gpu__; }
  void touchCPU() { m_dev = Device::__cpu__; }
};

template class XMux<ComplexMatrix>;
template class XMux<ComplexVector>;

template <typename V>
XMux<V> wrap_xmux(const V& obj) {
  XMux<V> res(obj);
  return res;
}