#include "culinear.h"

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void eig_gpu(const XMux<ComplexMatrix>& A, XMux<ComplexVector>& lambda,
             XMux<ComplexMatrix>& W) {
  A.to_gpu();
  lambda.to_gpu();
  W.to_gpu();

  auto& handle = CuHandleMgr::getInstance().getCuSolverHandle();
  cusolverDnParams_t params;
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  const cuComplex* d_pA = (const cuComplex*)A.device_data();
  cuComplex* d_A = const_cast<cuComplex*>(d_pA);

  int* d_info;
  int N = (int)A.getSize1();

  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  cuComplex* d_lam = (cuComplex*)lambda.device_data();
  cuComplex* d_W = (cuComplex*)W.device_data();

  size_t workspaceBytesOnDevice = 0;
  size_t workspaceBytesOnHost = 0;

  cudaDataType_t data_type = CUDA_C_32F;

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
}

void linsolve_gpu(const XMux<ComplexMatrix>& A, const XMux<ComplexVector>& b,
                  XMux<ComplexVector>& x) {
  XMux<ComplexMatrix> B(b.getSize(), 1);
  Complex* pB = B.cpu().getData();
  const Complex* pb = b.cpu().getData();
  memcpy(pB, pb, sizeof(Complex) * b.getSize());

  XMux<ComplexMatrix> X(b.getSize(), 1);
  linsolve_mat_gpu(A, B, X);

  Complex* d_X = X.device_data();
  if (!x.device_data()) {
    Complex* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(cuComplex) * b.getSize()));
    x.setDeviceData(d_x);
  }
  CUDA_CHECK(cudaMemcpy(x.device_data(), d_X, sizeof(cuComplex) * b.getSize(),
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
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(handle, params, n, n, CUDA_C_32F,
                                             d_A, n, CUDA_C_32F, &d_lwork,
                                             &h_lwork));

  void* d_work = nullptr;
  void* h_work = nullptr;
  if (d_lwork > 0) CUDA_CHECK(cudaMalloc(&d_work, d_lwork));
  if (h_lwork > 0) h_work = malloc(h_lwork);

  // factorization LU: A = P*L*U, detect singularity
  CUSOLVER_CHECK(cusolverDnXgetrf(handle, params, n, n, CUDA_C_32F, d_A, n,
                                  d_ipiv, CUDA_C_32F, d_work, d_lwork, h_work,
                                  h_lwork, d_info));

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
                                  CUDA_C_32F, d_A, n, d_ipiv, CUDA_C_32F, d_B,
                                  n, d_info));

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
  A.to_gpu(false);
  B.to_gpu(false);
  // solve A'X'=B'
  auto& handle = CuHandleMgr::getInstance().getCuSolverHandle();
  cusolverDnParams_t params = nullptr;

  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  void* d_A = A.device_data();
  void* d_B = B.device_data();
  int n = A.getSize1();
  int m = B.getSize1();

  int64_t* d_ipiv = nullptr;
  int* d_info = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int64_t) * n));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));

  size_t d_lwork = 0;
  size_t h_lwork = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(handle, params, n, n, CUDA_C_32F,
                                             d_A, n, CUDA_C_32F, &d_lwork,
                                             &h_lwork));
  void* d_work = nullptr;
  void* h_work = nullptr;
  if (d_lwork > 0) CUDA_CHECK(cudaMalloc(&d_work, d_lwork));
  if (h_lwork > 0) h_work = std::malloc(h_lwork);

  CUSOLVER_CHECK(cusolverDnXgetrf(handle, params, n, n, CUDA_C_32F, d_A, n,
                                  d_ipiv, CUDA_C_32F, d_work, d_lwork, h_work,
                                  h_lwork, d_info));

  int h_info = 0;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    std::cerr << "LU factorization failed: singlar matrix or error at "
              << h_info << std::endl;
  }

  // solve XA=B
  CUSOLVER_CHECK(cusolverDnXgetrs(
      handle, params, CUBLAS_OP_N,  // no op since already transpose A and B
      n, m, CUDA_C_32F, d_A, n, d_ipiv, CUDA_C_32F, d_B, n, d_info));

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

  transpose_gpu(n, m, (cuComplex*)d_B, (cuComplex*)d_B);
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

  const cuComplex* d_Ap = (cuComplex*)A.device_data();
  const cuComplex* d_vp = (cuComplex*)v.device_data();
  cuComplex* d_y = (cuComplex*)res.device_data();

  cuComplex* d_A = const_cast<cuComplex*>(d_Ap);
  cuComplex* d_v = const_cast<cuComplex*>(d_vp);

  auto& handle = CuHandleMgr::getInstance().getCuBlasHandle();
  CUBLAS_CHECK(cublasCgemv(handle, CUBLAS_OP_N, m, n, &CUCOMPLEX_ONE, d_A, m,
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

  const cuComplex* d_Ap = (cuComplex*)A.device_data();
  const cuComplex* d_Bp = (cuComplex*)B.device_data();
  cuComplex* d_C = (cuComplex*)res.device_data();

  cuComplex* d_A = const_cast<cuComplex*>(d_Ap);
  cuComplex* d_B = const_cast<cuComplex*>(d_Bp);

  auto& handle = CuHandleMgr::getInstance().getCuBlasHandle();

  CUBLAS_CHECK(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &CUCOMPLEX_ONE, d_A, m,  // leading dim of A
                           d_B, k,                  // leading dim of B
                           &CUCOMPLEX_ZERO, d_C, m  // leading dim of C
                           ));
  return res;
}