#include "culinear.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <cassert>

void eig_gpu(const XMux<ComplexMatrix>& A, XMux<ComplexVector>& lambda,
             XMux<ComplexMatrix>& W) {
  assert(lambda.getSize() > 0 && W.getSize() > 0);
  A.to_gpu();
  lambda.to_gpu();
  W.to_gpu();
  const int N = A.getSize1();

  auto& handle = CuHandleMgr::getInstance().getCuSolverHandle();
  cusolverDnParams_t params;
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  const CUDA_COMPLEX* d_pA = (const CUDA_COMPLEX*)A.device_data();
  CUDA_COMPLEX* d_A = const_cast<CUDA_COMPLEX*>(d_pA);

  int* d_info;

  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUDA_COMPLEX* d_lam = (CUDA_COMPLEX*)lambda.device_data();
  CUDA_COMPLEX* d_W = (CUDA_COMPLEX*)W.device_data();

  size_t workspaceBytesOnDevice = 0;
  size_t workspaceBytesOnHost = 0;

  cudaDataType_t data_type = CUDA_C_DATATYPE;

  CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params,
      CUSOLVER_EIG_MODE_NOVECTOR,               // no left eigenvector
      CUSOLVER_EIG_MODE_VECTOR,                 // right eigen vector
      N, data_type, d_A, N,                     // matrix
      data_type, d_lam, data_type, nullptr, N,  // left eigen vector
      data_type, d_W, N,                        // right eigen vector
      data_type, &workspaceBytesOnDevice, &workspaceBytesOnHost));

  void* d_work = nullptr;
  void* h_work = nullptr;
  if (workspaceBytesOnDevice > 0)
    CUDA_CHECK(cudaMalloc(&d_work, workspaceBytesOnDevice));
  if (workspaceBytesOnHost > 0) h_work = malloc(workspaceBytesOnHost);

  CUSOLVER_CHECK(
      cusolverDnXgeev(handle, params,
                      CUSOLVER_EIG_MODE_NOVECTOR,  // left eigen vector
                      CUSOLVER_EIG_MODE_VECTOR,    // right eigen vector
                      N, data_type, d_A, N,        // matrix
                      data_type, d_lam,            // eigen values
                      data_type, nullptr, N,       // left eigen vector
                      data_type, d_W, N,           // right eigen vector
                      data_type, d_work, workspaceBytesOnDevice,  //
                      h_work, workspaceBytesOnHost, d_info));

  CUDA_CHECK(cudaDeviceSynchronize());

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info) {
    std::cerr << "Failure in cusolver eig (Xgeev)" << std::endl;
  }

  cudaFree(d_info);
  cudaFree(d_work);
  free(h_work);
  CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  A.touchCPU();
}

void linsolve_gpu(const XMux<ComplexMatrix>& A, const XMux<ComplexVector>& b,
                  XMux<ComplexVector>& x) {
  A.to_gpu();
  b.to_gpu();
  if (x.getSize() == 0) {
    x = b;  // on gpu
    x.zero();
  }
  XMux<ComplexMatrix> B(b.getSize(), 1);
  CUDA_CHECK(cudaMemcpy(B.device_data(), b.device_data(),
                        sizeof(CUDA_COMPLEX) * b.getSize(),
                        cudaMemcpyDeviceToDevice));

  XMux<ComplexMatrix> X;
  linsolve_mat_gpu(A, B, X);

  void* d_X = X.device_data();
  CUDA_CHECK(cudaMemcpy(x.device_data(), d_X,
                        sizeof(CUDA_COMPLEX) * b.getSize(),
                        cudaMemcpyDeviceToDevice));
  x.touchGPU();
}

void linsolve_mat_gpu(const XMux<ComplexMatrix>& A,
                      const XMux<ComplexMatrix>& B, XMux<ComplexMatrix>& X) {
  XMux<ComplexMatrix> A_ = A;
  X = B;
  linsolve_inplace_gpu(A_, X);
}

void linsolve_inplace_gpu(XMux<ComplexMatrix>& A, XMux<ComplexMatrix>& B) {
  A.to_gpu();
  B.to_gpu();
  cusolverDnParams_t params = nullptr;

  auto& handle = CuHandleMgr::getInstance().getCuSolverHandle();
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  int n = A.getSize1();
  int nrhs = B.getSize2();

  void* d_A = (void*)A.device_data();
  void* d_B = (void*)B.device_data();

  int64_t* d_ipiv = nullptr;
  int* d_info = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int64_t) * n));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));

  // workspace query
  size_t d_lwork = 0, h_lwork = 0;
  CUSOLVER_CHECK(
      cusolverDnXgetrf_bufferSize(handle, params, n, n, CUDA_C_DATATYPE, d_A, n,
                                  CUDA_C_DATATYPE, &d_lwork, &h_lwork));

  void* d_work = nullptr;
  void* h_work = nullptr;
  if (d_lwork > 0) CUDA_CHECK(cudaMalloc(&d_work, d_lwork));
  if (h_lwork > 0) h_work = malloc(h_lwork);

  // factorization LU: A = P*L*U, detect singularity
  CUSOLVER_CHECK(cusolverDnXgetrf(handle, params, n, n, CUDA_C_DATATYPE, d_A, n,
                                  d_ipiv, CUDA_C_DATATYPE, d_work, d_lwork,
                                  h_work, h_lwork, d_info));

  int h_info = 0;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info > 0) {
    std::cerr << "LU factorization fails: Matrix U is singular at row: "
              << h_info << std::endl;
  } else if (h_info < 0) {
    std::cerr << "LU factorization fails: illegal value found at parameter: "
              << -h_info << std::endl;
  }

  // solve AX=B, B is overwritten with X
  CUSOLVER_CHECK(cusolverDnXgetrs(handle, params, CUBLAS_OP_N, n, nrhs,
                                  CUDA_C_DATATYPE, d_A, n, d_ipiv,
                                  CUDA_C_DATATYPE, d_B, n, d_info));

  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    std::cerr << "Linsolve failed: Info = " << h_info << std::endl;
  }

  if (d_ipiv) cudaFree(d_ipiv);
  if (d_info) cudaFree(d_info);
  if (d_work) cudaFree(d_work);
  if (h_work) free(h_work);
  if (params) cusolverDnDestroyParams(params);
}

void linsolve_right_inplace_gpu(XMux<ComplexMatrix>& A,
                                XMux<ComplexMatrix>& B) {
  A.to_gpu();
  B.to_gpu();
  // after this line, data is on gpu, column major
  void* d_AT;
  void* d_BT;
  CUDA_CHECK(cudaMalloc(&d_AT, sizeof(CUDA_COMPLEX) * A.getSize()));
  CUDA_CHECK(cudaMalloc(&d_BT, sizeof(CUDA_COMPLEX) * B.getSize()));
  transpose_gpu(A.getSize1(), A.getSize2(), (Complex*)A.device_data(),
                (Complex*)d_AT);
  transpose_gpu(B.getSize1(), B.getSize2(), (Complex*)B.device_data(),
                (Complex*)d_BT);
  // solving XA=B amounts to solving A'X'=B'
  // solve A'X'=B'
  auto& handle = CuHandleMgr::getInstance().getCuSolverHandle();
  cusolverDnParams_t params = nullptr;

  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  int n = A.getSize1();
  int m = B.getSize1();

  int64_t* d_ipiv = nullptr;
  int* d_info = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int64_t) * n));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));

  size_t d_lwork = 0;
  size_t h_lwork = 0;
  CUSOLVER_CHECK(
      cusolverDnXgetrf_bufferSize(handle, params, n, n, CUDA_C_DATATYPE, d_AT,
                                  n, CUDA_C_DATATYPE, &d_lwork, &h_lwork));
  void* d_work = nullptr;
  void* h_work = nullptr;
  if (d_lwork > 0) CUDA_CHECK(cudaMalloc(&d_work, d_lwork));
  if (h_lwork > 0) h_work = std::malloc(h_lwork);

  CUSOLVER_CHECK(cusolverDnXgetrf(handle, params, n, n, CUDA_C_DATATYPE, d_AT,
                                  n, d_ipiv, CUDA_C_DATATYPE, d_work, d_lwork,
                                  h_work, h_lwork, d_info));

  int h_info = 0;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    std::cerr << "LU factorization failed: singlar matrix or error at "
              << h_info << std::endl;
  }

  // solve XA=B
  CUSOLVER_CHECK(cusolverDnXgetrs(handle, params, CUBLAS_OP_N, n, m,
                                  CUDA_C_DATATYPE, d_AT, n, d_ipiv,
                                  CUDA_C_DATATYPE, d_BT, m, d_info));

  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info) {
    std::cerr << "linsolve failed with info " << h_info << std::endl;
  }

  // cleanup
  if (d_ipiv) cudaFree(d_ipiv);
  if (d_info) cudaFree(d_info);
  if (d_work) cudaFree(d_work);
  if (h_work) std::free(h_work);
  if (params) cusolverDnDestroyParams(params);

  transpose_gpu(n, m, (CUDA_COMPLEX*)d_BT, (CUDA_COMPLEX*)B.device_data());
}

void linsolve_right_gpu(const XMux<ComplexMatrix>& A,
                        const XMux<ComplexMatrix>& B, XMux<ComplexMatrix>& X) {
  XMux<ComplexMatrix> A_ = A;
  X = B;
  linsolve_right_inplace_gpu(A_, X);
}

XMux<ComplexVector> operator*(const XMux<ComplexMatrix>& A,
                              const XMux<ComplexVector>& v) {
  A.to_gpu();
  v.to_gpu();
  XMux<ComplexVector> res(v.getSize());
  //   res.to_gpu();

  int m = A.getSize1();
  int n = v.getSize();

  const CUDA_COMPLEX* d_Ap = (CUDA_COMPLEX*)A.device_data();
  const CUDA_COMPLEX* d_vp = (CUDA_COMPLEX*)v.device_data();
  CUDA_COMPLEX* d_y = (CUDA_COMPLEX*)res.device_data();

  CUDA_COMPLEX* d_A = const_cast<CUDA_COMPLEX*>(d_Ap);
  CUDA_COMPLEX* d_v = const_cast<CUDA_COMPLEX*>(d_vp);

  auto& handle = CuHandleMgr::getInstance().getCuBlasHandle();
  CUBLAS_CHECK(cublas_gemv(handle, CUBLAS_OP_N, m, n, &CUCOMPLEX_ONE, d_A, m,
                           d_v, 1, &CUCOMPLEX_ZERO, d_y, 1));
  return res;
}

XMux<ComplexMatrix> operator*(const XMux<ComplexMatrix>& A,
                              const XMux<ComplexMatrix>& B) {
  A.to_gpu();
  B.to_gpu();
  int m = A.getSize1();
  int n = B.getSize2();
  int k = A.getSize2();

  XMux<ComplexMatrix> res(m, n);
  //   res.to_gpu();

  const CUDA_COMPLEX* d_Ap = (CUDA_COMPLEX*)A.device_data();
  const CUDA_COMPLEX* d_Bp = (CUDA_COMPLEX*)B.device_data();
  CUDA_COMPLEX* d_C = (CUDA_COMPLEX*)res.device_data();

  CUDA_COMPLEX* d_A = const_cast<CUDA_COMPLEX*>(d_Ap);
  CUDA_COMPLEX* d_B = const_cast<CUDA_COMPLEX*>(d_Bp);

  auto& handle = CuHandleMgr::getInstance().getCuBlasHandle();

  CUBLAS_CHECK(cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &CUCOMPLEX_ONE, d_A, m,  // leading dim of A
                           d_B, k,                  // leading dim of B
                           &CUCOMPLEX_ZERO, d_C, m  // leading dim of C
                           ));
  cudaDeviceSynchronize();
  return res;
}

void fft2d(XMux<ComplexMatrix>& xa) {
  xa.to_gpu(false);

  auto& plan =
      CuHandleMgr::getInstance().getFFTPlan(xa.getSize1(), xa.getSize2());
  CUDA_COMPLEX* d_a = (CUDA_COMPLEX*)xa.device_data();

  cufftExec(plan, d_a, d_a, CUFFT_FORWARD);
}

void ifft2d(XMux<ComplexMatrix>& xa, Real s) {
  xa.to_gpu(false);
  auto& plan =
      CuHandleMgr::getInstance().getFFTPlan(xa.getSize1(), xa.getSize2());
  CUDA_COMPLEX* d_a = (CUDA_COMPLEX*)xa.device_data();
  cufftExec(plan, d_a, d_a, CUFFT_INVERSE);

  xa.scale(s);
}