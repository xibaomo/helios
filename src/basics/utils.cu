#include <cassert>
#include "utils.h"
#include "culinear.h"
__global__ void fftshift_2d_power2_kernel(const cuComplex* __restrict__ input,
                                          cuComplex* __restrict__ output,
                                          int width, int height) {
  int x_out = blockIdx.x * blockDim.x + threadIdx.x;
  int y_out = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_out < width && y_out < height) {
    // Bitwise XOR replacement for power-of-2 modulo arithmetic
    int x_in = x_out ^ (width >> 1);
    int y_in = y_out ^ (height >> 1);

    output[y_out * width + x_out] = input[y_in * width + x_in];
  }
}
__global__ void ifftshift_2d_pow2_kernel(const cuComplex* __restrict__ input,
                                         cuComplex* __restrict__ output,
                                         int width, int height) {
  // Calculate global thread coordinates (Row-Major: x = column, y = row)
  int x_out = blockIdx.x * blockDim.x + threadIdx.x;
  int y_out = blockIdx.y * blockDim.y + threadIdx.y;

  // Direct boundary check
  if (x_out < width && y_out < height) {
    // Bitwise optimization for Power of 2:
    // (x_out + width/2) % width is equivalent to x_out ^ (width / 2)
    int x_in = x_out ^ (width >> 1);
    int y_in = y_out ^ (height >> 1);

    // Linear index mapping (cuFFT Row-Major standard: y * width + x)
    int out_idx = y_out * width + x_out;
    int in_idx = y_in * width + x_in;

    // Coalesced memory transactions
    output[out_idx] = input[in_idx];
  }
}

void fftshift(XMux<ComplexMatrix>& A) {
  assert(isPower2(A.getSize1()) && isPower2(A.getSize2()));
  A.to_gpu();
  void* d_tmp;
  dim3 block_size(16, 16);

  dim3 grid_size((A.getSize1() + block_size.x - 1) / block_size.x,
                 (A.getSize2() + block_size.y - 1) / block_size.y);
  CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(ComplexMatrix::dtype) * A.getSize()));
  fftshift_2d_power2_kernel<<<grid_size, block_size>>>(
      (const cuComplex*)A.device_data(), (cuComplex*)d_tmp, A.getSize1(),
      A.getSize2());
  CUDA_CHECK(cudaMemcpy(A.device_data(), d_tmp,
                        sizeof(ComplexMatrix::dtype) * A.getSize(),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaFree(d_tmp));
  cudaDeviceSynchronize();
}

void ifftshift(XMux<ComplexMatrix>& A) {
  assert(isPower2(A.getSize1()) && isPower2(A.getSize2()));
  A.to_gpu();
  dim3 block_size(16, 16);

  dim3 grid_size((A.getSize1() + block_size.x - 1) / block_size.x,
                 (A.getSize2() + block_size.y - 1) / block_size.y);
  void* d_tmp;
  CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(ComplexMatrix::dtype) * A.getSize()));
  ifftshift_2d_pow2_kernel<<<grid_size, block_size>>>(
      (const cuComplex*)A.device_data(), (cuComplex*)d_tmp, A.getSize1(),
      A.getSize2());
  CUDA_CHECK(cudaMemcpy(A.device_data(), d_tmp,
                        sizeof(ComplexMatrix::dtype) * A.getSize(),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaFree(d_tmp));
  cudaDeviceSynchronize();
}

ComplexMatrix computeConvMat(const ComplexMatrix& eps_img,
                                     int max_order_x, int max_order_y) {
  size_t nx = eps_img.getSize1();
  size_t ny = eps_img.getSize2();
  ComplexMatrix F_eps = eps_img;
  auto mx_F_eps = wrap_xmux(F_eps);
  fft2d(mx_F_eps);
  mx_F_eps.scale(1.f / (nx * ny));
  fftshift(mx_F_eps);
  mx_F_eps.to_cpu();

  int kx_min = -nx/2;
  int ky_min = -ny/2;

  int orders_x = 2 * max_order_x + 1;
  int orders_y = 2 * max_order_y + 1;
  int total_orders = orders_x * orders_y;
  ComplexMatrix conv_mat(total_orders, total_orders);
  conv_mat.zero();
  for (size_t i = 0; i < total_orders; i++) {
    int kx_out = i / orders_y;
    int ky_out = i % orders_y;
    for (size_t j = 0; j < total_orders; j++) {
      int kx_in = j / orders_y;
      int ky_in = j % orders_y;

      int idx_kx = (kx_out - kx_in) - kx_min;
      int idx_ky = (ky_out - ky_in) - ky_min;

      conv_mat[i][j] = F_eps[idx_kx][idx_ky];

    }
  }

  return conv_mat;
}