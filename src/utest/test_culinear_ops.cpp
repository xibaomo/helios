#include "arr.h"
#include "cuhandlemgr.h"
#include "culinear.h"
#include "utest.h"
#include "utils.h"
#include "xmux.h"
#define N 4
using namespace std;

static bool test_transpose();
static AddUnitTest t_trans("test_transpose", test_transpose);
static bool test_eig();
static AddUnitTest t_eig("test_eig", test_eig);
static bool test_linsolve();
static AddUnitTest t_linsolve("test_linsolve", test_linsolve);
static bool test_linsolve_vector();
static AddUnitTest t_linsolvevec("test_linsolve_vec", test_linsolve_vector);
static bool test_linsolve_right();
static AddUnitTest t_linsolve_right("test_linsolve_right", test_linsolve_right);
static bool test_matvec();
static AddUnitTest t_matvec("test_matvec", test_matvec);
static bool test_matmul();
static AddUnitTest t_matmul("test_matmul", test_matmul);
static bool test_fft2d();
static AddUnitTest t_fft2d("test_fft2d", test_fft2d);
static bool test_convmat();
static AddUnitTest t_convmat("test_convmat",test_convmat);

static ComplexMatrix createMat(size_t N1, size_t N2,
                               Complex s = Complex{1.f, 0.f}) {
  ComplexMatrix a(N1, N2);
  Complex* p = a.getData();
  for (size_t i = 0; i < N1 * N2; i++) {
    p[i] = Complex{i + 0.f, 0.f} + s;
  }
  return a;
}

static ComplexVector createVector(size_t n,
                                  Complex s = Complex{N * N + 1.f, 0.f}) {
  ComplexVector v(n);
  for (size_t i = 0; i < v.getSize(); i++) {
    v[i] = Complex{i + 0.f, 0.f} + s;
  }
  return v;
}

bool test_transpose() {
  ComplexMatrix a(2, 3);
  Complex* p = a.getData();
  for (size_t i = 0; i < a.getSize(); i++) {
    p[i] = Complex{i + 1.f, 0.f};
  }

  auto xa = wrap_xmux(a);
  xa.to_gpu();

  //   xa.to_cpu(false);
  xa.to_cpu();
  for (size_t i = 0; i < a.getSize(); i++) {
    cout << p[i] << " ";
  }
  cout << endl;

  return true;
}

bool test_eig() {
  ComplexMatrix a = createMat(N, N);

  ComplexVector lam(N);
  ComplexMatrix W(N, N);

  auto xa = wrap_xmux(a);
  auto xl = wrap_xmux(lam);
  auto xw = wrap_xmux(W);

  eig_gpu(xa, xl, xw);
  xl.to_cpu();
  xw.to_cpu();

  show_arr(lam);
  show_arr(W);

  return true;
}

bool test_linsolve() {
  ComplexMatrix a = createMat(N, N);
  ComplexVector v = createVector(N);
  ComplexMatrix b = createMat(N, N, N * N + 1.f);
  ComplexMatrix X = createMat(N, N);

  auto xa = wrap_xmux(a);
  auto xb = wrap_xmux(b);
  auto xv = wrap_xmux(v);
  auto xx = wrap_xmux(X);

  cout << "A: " << endl;
  show_arr(a);
  cout << "B: " << endl;
  show_arr(b);

  linsolve_mat_gpu(xa, xb, xx);
  xx.to_cpu();
  cout << "X: " << endl;
  show_arr(X);

  XMux<ComplexMatrix> b_res = xa * xx;
  cout << "AX: " << endl;
  show_arr(b_res.cpu());

  return true;
}
bool test_linsolve_vector() {
  ComplexMatrix a = createMat(N, N);
  ComplexVector v = createVector(N);
  ComplexVector X = createVector(N);

  auto xa = wrap_xmux(a);
  auto xv = wrap_xmux(v);
  auto xx = wrap_xmux(X);

  cout << "A: " << endl;
  show_arr(a);
  cout << "B: " << endl;
  show_arr(v);

  linsolve_gpu(xa, xv, xx);
  xx.to_cpu();
  cout << "X: " << endl;
  show_arr(X);

  XMux<ComplexVector> b = xa * xx;
  cout << "AX: " << endl;
  show_arr(b.cpu());

  return true;
}

bool test_linsolve_right() {
  ComplexMatrix a = createMat(N, N);
  ComplexMatrix b = createMat(N, N, N * N + 1.f);
  ComplexMatrix x = createMat(N, N);

  auto xa = wrap_xmux(a);
  auto xb = wrap_xmux(b);
  auto xx = wrap_xmux(x);

  cout << "A: " << endl;
  show_arr(a);
  cout << "B: " << endl;
  show_arr(b);
  linsolve_right_gpu(xa, xb, xx);
  xx.to_cpu();
  cout << "X=BA^-1: " << endl;
  show_arr(x);

  XMux<ComplexMatrix> b_res = xx * xa;
  cout << "XA: " << endl;
  show_arr(b_res.cpu());
  b.for_each([](Complex& a, Complex& b) { return a - b; }, b_res.cpu());
  if (b.norm() > 1.e-4) return false;

  return true;
}

bool test_matvec() {
  ComplexMatrix a = createMat(N, N);
  ComplexVector b = createVector(N, N * N + 1.f);

  auto xa = wrap_xmux(a);
  auto xb = wrap_xmux(b);
  //   auto xx = wrap_xmux(x);

  cout << "A: " << endl;
  show_arr(a);
  cout << "B: " << endl;
  show_arr(b);
  XMux<ComplexVector> xx = xa * xb;
  xx.to_cpu();
  cout << "a*b: " << endl;
  show_arr(xx.cpu());
  return true;
}

bool test_matmul() {
  ComplexMatrix a = createMat(N, N);
  ComplexMatrix b = createMat(N, N, N * N + 1.f);

  auto xa = wrap_xmux(a);
  auto xb = wrap_xmux(b);
  //   auto xx = wrap_xmux(x);

  cout << "A: " << endl;
  show_arr(a);
  cout << "B: " << endl;
  show_arr(b);
  XMux<ComplexMatrix> xx = xa * xb;
  xx.to_cpu();
  cout << "a*b: " << endl;
  show_arr(xx.cpu());
  return true;
}

bool test_fft2d() {
  ComplexMatrix a = createMat(N, N);
  auto xa = wrap_xmux(a);

  cout << "A: " << endl;
  show_arr2d(a);
  fft2d(xa);
  fftshift(xa);
  xa.to_cpu();
  cout << "fft2d: " << endl;
  show_arr2d(xa.cpu());

  ifftshift(xa);
  ifft2d(xa, 1.f / (N * N));
  cout << "ifft2d: " << endl;
  show_arr2d(xa.cpu());

  return true;
}

bool test_convmat() {
  ComplexMatrix a = createMat(N*2,N*2);
  a.for_each([](auto& x) { return x+ Complex{0.f, 1.f};});
  // auto xa = wrap_xmux(a);
  cout << "A: " << endl;
  show_arr2d(a);

  ComplexMatrix cm = computeConvMat(a, 1, 1);
  cout << "conv mat: " << endl;
  show_arr2d(cm);
  return true;
}