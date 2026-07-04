#pragma once
#include <cuComplex.h>
#include <library_types.h>
#include <cufft.h>

#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

// #define USE_32_BIT

#ifdef USE_32_BIT
typedef float Real;
inline const cudaDataType_t CUDA_C_DATATYPE = CUDA_C_32F;
inline const cufftType_t CUFFT_TYPE = CUFFT_C2C;
typedef cuComplex CUDA_COMPLEX;
#define make_cuda_complex make_cuComplex
#define cublas_geam cublasCgeam
#define cublas_gemv cublasCgemv
#define cublas_gemm cublasCgemm
#define cufftExec cufftExecC2C
#define cu_abs cuCabsf
#define cu_mul cuCmulf
#define cu_add cuCaddf
#define cu_sub cuCsubf
#define cu_real cuCrealf
#define cu_imag cuCimagf
#else
typedef double Real;
inline const cudaDataType_t CUDA_C_DATATYPE = CUDA_C_64F;
inline const cufftType_t CUFFT_TYPE = CUFFT_Z2Z;
typedef cuDoubleComplex CUDA_COMPLEX;
#define make_cuda_complex make_cuDoubleComplex
#define cublas_geam cublasZgeam
#define cublas_gemv cublasZgemv
#define cublas_gemm cublasZgemm
#define cufftExec cufftExecZ2Z
#define cu_abs cuCabs
#define cu_mul cuCmul
#define cu_add cuCadd
#define cu_sub cuCsub
#define cu_real cuCreal
#define cu_imag cuCimag
#endif

typedef std::string String;
typedef std::complex<Real> Complex;

#define JJ Complex{0.f, 1.f}

inline const Complex COMPLEX_ONE = Complex{1.f, 0.f};
inline const Complex COMPLEX_ZERO = Complex{0.f, 0.f};
inline const CUDA_COMPLEX CUCOMPLEX_ONE = make_cuda_complex(1.f, 0.f);
inline const CUDA_COMPLEX CUCOMPLEX_ZERO = make_cuda_complex(0.f, 0.f);