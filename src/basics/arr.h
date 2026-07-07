#pragma once
#include <complex>
#include <cstring>
#include <memory>

#include "types.h"

template <typename T>
class Array1D {
 protected:
  std::unique_ptr<T[]> m_data = nullptr;
  size_t m_size = 0;

 public:
  using dtype = T;
  Array1D(size_t s = 0) : m_data(nullptr), m_size(s) {
    resize(m_size);
    zero();
  }

  // copy constructor
  Array1D(const Array1D<T>& other) {
    m_size = other.m_size;
    if (m_size == 0) return;

    m_data = std::make_unique<T[]>(m_size);
    std::copy(other.m_data.get(), other.m_data.get() + m_size, m_data.get());
  }
  // copy assignment
  Array1D<T>& operator=(const Array1D<T>& other) {
    if (m_data && m_size != other.m_size) {
      m_data.reset();
    }
    m_size = other.m_size;
    if (m_size > 0) {
      m_data = std::make_unique<T[]>(m_size);
      std::copy(other.m_data.get(), other.m_data.get() + m_size, m_data.get());
    }
    return *this;
  }
  // move constructor
  Array1D(Array1D<T>&& other) {
    m_data = std::move(other.m_data);
    m_size = other.m_size;
    other.m_size = 0;
    other.m_data = nullptr;
  }
  // move assignment
  Array1D<T>& operator=(Array1D<T>&& other) {
    if (m_data) {
      m_data.reset();
    }
    m_data = std::move(other.m_data);
    m_size = other.m_size;
    other.m_size = 0;
    other.m_data = nullptr;
    return *this;
  }

  void resize(size_t s) {
    if (s != m_size) m_data.reset();
    m_size = s;
    if (s > 0 && !m_data) {
      m_data = std::make_unique<T[]>(s);
    }
  }

  void zero() {
    if (m_size > 0) {
      memset(m_data.get(), 0, sizeof(T) * m_size);
    }
  }

  T* getData() { return m_data.get(); }
  const T* getData() const { return m_data.get(); }

  size_t getSize() const { return m_size; }

  T& operator[](size_t i) { return m_data[i]; }
  const T& operator[](size_t i) const { return m_data[i]; }

  template <typename F, typename... Args>
  void for_each(F fn, Array1D<Args>&... others) {
    for (size_t i = 0; i < m_size; i++) {
      m_data[i] = fn(m_data[i], others[i]...);
    }
  }

  Real norm() const {
    Real s = 0.f;
    for (size_t i = 0; i < m_size; i++) {
      s += abs(m_data[i]) * abs(m_data[i]);
    }
    return std::sqrt(s);
  }

  Array1D<T> getSubArray(size_t start, size_t sz) {
    Array1D<T> ret(sz);
    size_t k = 0;
    for (size_t i = start; i < start + sz; i++) {
      ret[k++] = m_data[i];
    }
    return ret;
  }

  T& end() { return m_data[m_size - 1]; }

  __attribute__((noinline)) T sum() {
    T s = 0.f;
    for (size_t i = 0; i < m_size; i++) s += m_data[i];
    return s;
  }

  __attribute__((noinline)) Real mean_abs2() const {
    Real s = 0.f;
    for (size_t i = 0; i < m_size; i++) {
      Real a = std::abs(m_data.get()[i]);
      s += a * a;
    }
    if (s < -1.0f) {
      std::cout << "This will never happen but forces symbol retention: " << s
                << std::endl;
    }
    return s / m_size;
  }
};

template class Array1D<int>;
template class Array1D<float>;
template class Array1D<double>;
template class Array1D<std::complex<float>>;
template class Array1D<std::complex<double>>;
// *****************************************************************************

template <typename T>
class Array2D : public Array1D<T> {
 protected:
  size_t m_size1, m_size2;

 public:
  Array2D(size_t s1 = 0, size_t s2 = 0)
      : Array1D<T>(s1 * s2), m_size1(s1), m_size2(s2) {}

  void resize(size_t s1, size_t s2) {
    m_size1 = s1;
    m_size2 = s2;
    Array1D<T>::resize(s1 * s2);
    Array1D<T>::zero();
  }

  Array2D(const Array2D<T>& other)
      : Array1D<T>(other), m_size1(other.m_size1), m_size2(other.m_size2) {}

  // copy assignment
  Array2D& operator=(const Array2D<T>& other) {
    if (this != &other) {
      Array1D<T>::operator=(other);
      m_size1 = other.m_size1;
      m_size2 = other.m_size2;
    }
    return *this;
  }

  // move constructor
  Array2D(Array2D<T>&& other) noexcept
      : Array1D<T>(std::move(other)),
        m_size1(other.m_size1),
        m_size2(other.m_size2) {
    other.m_size1 = 0;
    other.m_size2 = 0;
  }

  // move assignment
  Array2D& operator=(Array2D<T>&& other) noexcept {
    if (this != &other) {
      Array1D<T>::operator=(std::move(other));
      m_size1 = other.m_size1;
      m_size2 = other.m_size2;
      other.m_size1 = 0;
      other.m_size2 = 0;
    }
    return *this;
  }

  T* operator[](size_t i) { return this->m_data.get() + m_size2 * i; }
  const T* operator[](size_t i) const {
    return this->m_data.get() + m_size2 * i;
  }

  size_t getSize1() const { return m_size1; }
  size_t getSize2() const { return m_size2; }

  void eye() {
    Array1D<T>::zero();
    for (size_t i = 0; i < m_size1; i++) {
      (*this)[i][i] = 1.f;
    }
  }
};

template class Array2D<int>;
template class Array2D<float>;
template class Array2D<Complex>;

typedef Array2D<Complex> ComplexMatrix;
typedef Array1D<Complex> ComplexVector;

typedef Array2D<Real> RealMatrix;
typedef Array1D<Real> RealVector;