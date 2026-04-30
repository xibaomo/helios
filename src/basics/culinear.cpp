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