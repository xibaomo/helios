#pragma once
#include <iostream>
#include "xmux.h"

inline bool isPower2(int n) {
    return n > 0 && (n & (n-1)) == 0;
}

void fftshift(XMux<ComplexMatrix>& A);
void ifftshift(XMux<ComplexMatrix>& A);

ComplexMatrix computeConvMat(const ComplexMatrix& a, int max_order_x, int max_order_y);

template <typename T>
static void show_arr(const T& arr) {
  const typename T::dtype* p = arr.getData();
  for (size_t i = 0; i < arr.getSize(); i++) {
    std::cout << p[i] << " ";
  }
  std::cout << std::endl;
}
template <typename T>
static void show_arr2d(const T& arr) {
  for (size_t i = 0; i < arr.getSize1(); i++) {
    for (size_t j = 0; j < arr.getSize2(); j++) {
      std::cout << arr[i][j] << "\t";
    }
    std::cout << std::endl;
  }
}