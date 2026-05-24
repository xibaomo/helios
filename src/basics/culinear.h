#pragma once

#include "arr.h"
#include "xmux.h"

void eig_gpu(const XMux<ComplexMatrix>& A, XMux<ComplexVector>& lambda,
             XMux<ComplexMatrix>& W);

void linsolve_mat_gpu(const XMux<ComplexMatrix>& A,
                      const XMux<ComplexMatrix>& B, XMux<ComplexMatrix>& X);

void linsolve_gpu(const XMux<ComplexMatrix>& A, const XMux<ComplexVector>& b,
                  XMux<ComplexVector>& x);

XMux<ComplexVector> operator*(const XMux<ComplexMatrix>& A,
                              const XMux<ComplexVector>& v);
XMux<ComplexMatrix> operator*(const XMux<ComplexMatrix>& A,
                              const XMux<ComplexMatrix>& B);
// XMux<ComplexMatrix> operator+(const XMux<ComplexMatrix>& A,
//                               const XMux<ComplexMatrix>& B);

void linsolve_inplace_gpu(XMux<ComplexMatrix>& A, XMux<ComplexMatrix>& B);

// solve XA=B, X=BA^-1
void linsolve_right_inplace_gpu(XMux<ComplexMatrix>& A, XMux<ComplexMatrix>& B);
void linsolve_right_gpu(const XMux<ComplexMatrix>& A,
                        const XMux<ComplexMatrix>& B, XMux<ComplexMatrix>& X);

// fft-related
void fft2d(XMux<ComplexMatrix>& a);
void ifft2d(XMux<ComplexMatrix>& a, Real s = 1.f);
