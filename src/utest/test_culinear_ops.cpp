#include "arr.h"
#include "cuhandlemgr.h"
#include "culinear.h"
#include "utest.h"
#include "xmux.h"
#define N 2
using namespace std;

static bool test_transpose();
static AddUnitTest t_trans("test_transpose", test_transpose);
static bool test_eig();
static AddUnitTest t_eig("test_eig", test_eig);
static bool test_linsolve();
static AddUnitTest t_linsolve("test_linsolve", test_linsolve);

static ComplexMatrix createMat(size_t N1, size_t N2, Complex s= Complex{1.f,0.f}) {
  ComplexMatrix a(N1, N2);
  Complex* p = a.getData();
  for (size_t i = 0; i < N1 * N2; i++) {
    p[i] = Complex{i + 0.f, 0.f} + s;
  }
  return a;
}

static ComplexVector createVector(size_t n, Complex s = Complex{N*N+1.f,0.f}){
    ComplexVector v(n);
    for(size_t i=0; i < v.getSize(); i++) {
        v[i] = Complex{i+0.f,0.f} + s;
    }
    return v;
}

template <typename T>
static void show_arr(const T& arr) {
  const typename T::dtype* p = arr.getData();
  for (size_t i = 0; i < arr.getSize(); i++) {
    cout << p[i] << " ";
  }
  cout << endl;
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
    ComplexMatrix a = createMat(N,N);
    ComplexVector v = createVector(N);
    ComplexMatrix b = createMat(N,N,N*N+1.f);
    ComplexMatrix X = createMat(N,N);

    auto xa = wrap_xmux(a);
    auto xb = wrap_xmux(b);
    auto xv = wrap_xmux(v);
    auto xx = wrap_xmux(X);

    cout << "A: " << endl;
    show_arr(a);
    cout << "B: " << endl;
    show_arr(b);

    linsolve_mat_gpu(xa,xb,xx);
    xx.to_cpu();
    cout << "X: " << endl;
    show_arr(X);

    return true;
}
