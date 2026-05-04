#pragma once
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <iostream>
#include <memory>

#define CUDA_CHECK(err)                                                    \
  {                                                                        \
    if (err != cudaSuccess) {                                              \
      std::cerr << "Cuda Error: " << cudaGetErrorString(err) << std::endl; \
      exit(-1);                                                            \
    }                                                                      \
  }

#define CUSOLVER_CHECK(err)                                     \
  {                                                             \
    if (err != CUSOLVER_STATUS_SUCCESS) {                       \
      std::cerr << "cuSolver Error: " << (int)err << std::endl; \
      exit(-1);                                                 \
    }                                                           \
  }

#define CUBLAS_CHECK(err)                                     \
  {                                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                       \
      std::cerr << "cuBLAS error: " << (int)err << std::endl; \
      exit(-1);                                               \
    }                                                         \
  }

class CuHandleMgr {
 private:
  cublasHandle_t m_cublas_handle;
  cusolverDnHandle_t m_cusolver_handle;
  typedef std::pair<size_t, size_t> PlanKey;
  std::map<PlanKey, cufftHandle> m_fftPlans;
  CuHandleMgr() {
    CUBLAS_CHECK(cublasCreate(&m_cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&m_cusolver_handle));
  }

 public:
  static CuHandleMgr& getInstance() {
    static CuHandleMgr _ins;
    return _ins;
  }

  ~CuHandleMgr() {
    CUBLAS_CHECK(cublasDestroy(m_cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(m_cusolver_handle));

    for(auto& it : m_fftPlans){
        cufftDestroy(it.second);
    }
  }
  cublasHandle_t& getCuBlasHandle() { return m_cublas_handle; }
  cusolverDnHandle_t& getCuSolverHandle() { return m_cusolver_handle; }

  cufftHandle& getFFTPlan(size_t n1, size_t n2) {
    PlanKey key{n1, n2};
    try {
      auto& h = m_fftPlans.at(key);
      return h;
    } catch (...) {
      cufftHandle plan;
      cufftPlan2d(&plan, n1, n2, CUFFT_C2C);
      m_fftPlans[key] = std::move(plan);
    }
    return m_fftPlans[key];
  }
};