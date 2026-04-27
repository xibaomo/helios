#pragma once
#include <complex>
#include <cstring>
#include <memory>
template <typename T>
class Array2D {
 protected:
  std::unique_ptr<T[]> m_data;
  size_t m_size1, m_size2, m_size;

 public:
  Array2D(size_t s1 = 0, size_t s2 = 0)
      : m_data(nullptr), m_size1(s1), m_size2(s2) {
    resize(m_size1, m_size2);
  }

  void resize(size_t s1, size_t s2) {
    m_size = s1 * s2;
    if (m_size > 0) {
      m_data = std::make_unique<T[]>(m_size);
    }
  }

  void zero() {
    if (m_size > 0) {
      memset(m_data.get(), 0, sizeof(T) * m_size);
    }
  }
};

typedef std::complex<float> Complex;

template class Array2D<int>;
template class Array2D<float>;
template class Array2D<Complex>;
