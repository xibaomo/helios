#include <cassert>

#include "culinear.h"
#include "utils.h"

// CUDA Kernel for 2D FFTShift (Optimized for Column-Major Layout)
__global__ void fftshift_2d_col_major_kernel(const cuComplex* __restrict__ d_in,
                                             cuComplex* __restrict__ d_out,
                                             int H, int W) {
  // CRITICAL: threadIdx.x must map to the Row (fastest-changing index in
  // memory)
  int out_row = blockIdx.x * blockDim.x + threadIdx.x;
  int out_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check for the output grid
  if (out_row < H && out_col < W) {
    // 1. Calculate shift offsets for both axes
    // For fftshift: shift = (N + 1) / 2
    // For ifftshift: shift = N / 2
    int shift_row = (H + 1) / 2;
    int shift_col = (W + 1) / 2;

    // 2. Map Output index back to Input index periodically
    // (out - shift + N) % N guarantees correct mathematical wrapping
    int in_row = (out_row - shift_row + H) % H;
    int in_col = (out_col - shift_col + W) % W;

    // 3. Compute flat linear memory addresses using Column-Major formula: (col
    // * H + row)
    int out_idx = out_col * H + out_row;
    int in_idx = in_col * H + in_row;

    // 4. Global memory streaming (Fully Coalesced because out_row is bound to
    // threadIdx.x)
    d_out[out_idx] = d_in[in_idx];
  }
}
void fftshift(XMux<ComplexMatrix>& A) {
  A.to_gpu();
  void* d_out;
  void* d_in = A.device_data();
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(cuComplex) * A.getSize()));
  dim3 blockDim(32, 16);
  int H = A.getSize1();
  int W = A.getSize2();

  // Grid bounds calculated using Row-first logic
  dim3 gridDim((H + blockDim.x - 1) / blockDim.x,
               (W + blockDim.y - 1) / blockDim.y);

  // Launch the Out-of-Place shift kernel
  fftshift_2d_col_major_kernel<<<gridDim, blockDim, 0, 0>>>(
      (cuComplex*)d_in, (cuComplex*)d_out, H, W);

  // Synchronous/Asynchronous error checking
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Column-Major FFTShift launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }

  CUDA_CHECK(cudaMemcpy(d_in, d_out, sizeof(cuComplex) * A.getSize(),
                        cudaMemcpyDeviceToDevice));
  cudaFree(d_out);

  cudaDeviceSynchronize();
}

__global__ void ifftshift_2d_col_major_kernel(
    const cuComplex* __restrict__ d_in, cuComplex* __restrict__ d_out, int H,
    int W) {
  // Coalescing Rule: threadIdx.x must handle the fastest-changing memory
  // dimension. In Column-Major, this is the Row index.
  int out_row = blockIdx.x * blockDim.x + threadIdx.x;
  int out_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Grid boundary check
  if (out_row < H && out_col < W) {
    // 1. Calculate shift steps specifically for IFFTShift
    // For ifftshift, the step size is exactly N / 2
    int shift_row = H / 2;
    int shift_col = W / 2;

    // 2. Map Output index back to Input index
    // (out + shift) % N wraps around periodically
    int in_row = (out_row + shift_row) % H;
    int in_col = (out_col + shift_col) % W;

    // 3. Compute flat 1D addresses using Column-Major formula: (col * H + row)
    int out_idx = out_col * H + out_row;
    int in_idx = in_col * H + in_row;

    // 4. Stream data through global memory lines
    // Fully coalesced because out_row is directly tied to threadIdx.x
    d_out[out_idx] = d_in[in_idx];
  }
}

void ifftshift(XMux<ComplexMatrix>& A) {
  assert(isPower2(A.getSize1()) && isPower2(A.getSize2()));
  A.to_gpu();
  dim3 block_size(16, 16);

  dim3 grid_size((A.getSize1() + block_size.x - 1) / block_size.x,
                 (A.getSize2() + block_size.y - 1) / block_size.y);
  void* d_tmp;
  CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(ComplexMatrix::dtype) * A.getSize()));
  ifftshift_2d_col_major_kernel<<<grid_size, block_size>>>(
      (const cuComplex*)A.device_data(), (cuComplex*)d_tmp, A.getSize1(),
      A.getSize2());
  CUDA_CHECK(cudaMemcpy(A.device_data(), d_tmp,
                        sizeof(ComplexMatrix::dtype) * A.getSize(),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaFree(d_tmp));
  cudaDeviceSynchronize();
}

ComplexMatrix computeConvMat(const ComplexMatrix& eps_img, int max_order_x,
                             int max_order_y) {
  size_t nx = eps_img.getSize1();
  size_t ny = eps_img.getSize2();
  ComplexMatrix F_eps = eps_img;
  auto mx_F_eps = wrap_xmux(F_eps);
  fft2d(mx_F_eps);
  mx_F_eps.scale(1.f / (nx * ny));
  fftshift(mx_F_eps);
  mx_F_eps.to_cpu();

  int kx_min = -nx / 2;
  int ky_min = -ny / 2;

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

// Structure to store 2D plane vectors
struct Real2 {
  Real x;  // X-component (corresponds to changes across columns)
  Real y;  // Y-component (corresponds to changes across rows)
};

// ============================================================================
// CUDA Kernel: Computes 2D normalized gradient direction field using Sobel
// Layout: Optimized for Column-Major matrix memory alignment
// ============================================================================
__global__ void compute_gradient_2d_sobel_kernel(
    const cuComplex* __restrict__ d_in, Real2* __restrict__ d_gradients, int H,
    int W, Real dx, Real dy) {
  // Coalescing Rule: threadIdx.x must handle the contiguous row index (i)
  // to achieve perfect memory coalescing in Column-Major layout.
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index (Y-geometry)
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index (X-geometry)

  if (i < H && j < W) {
    // Boundary-safe lambda to compute complex magnitude securely at clamped
    // edges
    auto get_height = [&](int r, int c) -> Real {
      r = max(0, min(r, H - 1));
      c = max(0, min(c, W - 1));
      return cuCabsf(
          d_in[c * H + r]);  // Column-Major flat indexing: col * H + row
    };

    // ====================================================================
    // 1. Apply Sobel X-Operator in index space, then scale by physical dx
    // ====================================================================
    Real h00 = get_height(i - 1, j - 1);
    Real h10 = get_height(i, j - 1);
    Real h20 = get_height(i + 1, j - 1);

    Real h02 = get_height(i - 1, j + 1);
    Real h12 = get_height(i, j + 1);
    Real h22 = get_height(i + 1, j + 1);

    // Compute physical gradient along X-axis (divided by 8 for operator
    // normalization)
    Real dg_dx =
        ((h02 + 2.0f * h12 + h22) - (h00 + 2.0f * h10 + h20)) / (8.0f * dx);

    // ====================================================================
    // 2. Apply Sobel Y-Operator in index space, then scale by physical dy
    // ====================================================================
    Real h01 = get_height(i - 1, j);
    Real h21 = get_height(i + 1, j);

    // Compute physical gradient along Y-axis
    Real dg_dy =
        ((h20 + 2.0f * h21 + h22) - (h00 + 2.0f * h01 + h02)) / (8.0f * dy);

    // ====================================================================
    // 3. Vector magnitude calculation and 2D normalization
    // ====================================================================
    Real len_2d = sqrtf(dg_dx * dg_dx + dg_dy * dg_dy);

    int out_idx = j * H + i;  // Destination index in Column-Major output

    // Normalize to get pure directional vectors and prevent division by zero
    if (len_2d > 1e-8f) {
      Real inv_len = 1.0f / len_2d;
      d_gradients[out_idx].x = dg_dx * inv_len;
      d_gradients[out_idx].y = dg_dy * inv_len;
    } else {
      // Flat regions have no explicit gradient direction
      d_gradients[out_idx].x = 0.0f;
      d_gradients[out_idx].y = 0.0f;
    }
  }
}

std::tuple<RealMatrix, RealMatrix> generateNormalField(
    const ComplexMatrix& eps_img, Real dx, Real dy) {
  auto xm_eps_img = wrap_xmux(eps_img);
  xm_eps_img.to_gpu();

  std::tuple<RealMatrix, RealMatrix> normal_field;
  // 32 threads in X map to continuous row mrmory, ensure Warp Coalescing
  dim3 blockDim(32, 16);

  const int H = eps_img.getSize1();
  const int W = eps_img.getSize2();

  dim3 gridDim((H + blockDim.x - 1) / blockDim.x,
               (W + blockDim.y - 1) / blockDim.y);

  void* d_in = xm_eps_img.device_data();
  void* d_normals;
  CUDA_CHECK(cudaMalloc(&d_normals, sizeof(Real2) * H * W));

  compute_gradient_2d_sobel_kernel<<<gridDim, blockDim, 0, 0>>>(
      (cuComplex*)d_in, (Real2*)d_normals, H, W, dx, dy);

  // ??????
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Sobel normal field kernel failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  Real2* h_normals = (Real2*)malloc(sizeof(Real2) * H * W);
  CUDA_CHECK(cudaMemcpy(h_normals, d_normals, sizeof(Real2) * H * W,
                        cudaMemcpyDeviceToHost));
  cudaFree(d_normals);

  auto& [nx, ny] = normal_field;
  nx.resize(H, W);
  ny.resize(H, W);
  for (size_t i = 0; i < H; i++) {
    for (size_t j = 0; j < W; j++) {
      nx[i][j] = h_normals[i * W + j].x;
      ny[i][j] = h_normals[i * W + j].y;
    }
  }

  return normal_field;
}