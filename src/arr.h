#pragma once
#include <complex>
#include <cstring>
#include <memory>

template <typename T>
class Array1D {
 protected:
  std::unique_ptr<T[]> m_data;
  size_t m_size;

 public:
  Array1D(size_t s = 0) : m_data(nullptr), m_size(s) {
    resize(m_size);
    zero();
  }

  void resize(size_t s) {
    if (s > 0) {
      m_data = std::make_unique<T[]>(s);
    }
  }

  void zero() {
    if (m_size > 0) {
      memset(m_data.get(), 0, sizeof(T) * m_size);
    }
  }
};

template <typename T>
class Array2D : public Array1D<T> {
 protected:
  size_t m_size1, m_size2;

 public:
  Array2D(size_t s1 = 0, size_t s2 = 0)
      : Array1D<T>(s1 * s2), m_size1(s1), m_size2(s2) {}
};
typedef std::complex<float> Complex;

template class Array2D<int>;
template class Array2D<float>;
template class Array2D<Complex>;
template class Array1D<Complex>;

typedef Array2D<Complex> ComplexMatrix;
typedef Array1D<Complex> ComplexVector;