#include "arr.h"
#include "cuhandlemgr.h"
#include "utest.h"
#include "xmux.h"
#include "culinear.h"
using namespace std;

static bool test_transpose();
static AddUnitTest t_trans("test_transpose", test_transpose);
static bool test_eig();
static AddUnitTest t_eig("test_eig", test_eig);

static ComplexMatrix createMat(size_t N1, size_t N2) {
  ComplexMatrix a(N1, N2);
  Complex* p = a.getData();
  for (size_t i = 0; i < N1 * N2; i++) {
    p[i] = Complex{i + 1.f, 0.f};
  }
  return a;
}

template <typename T>
static void show_arr(const T& arr) {
    const typename T::dtype* p=arr.getData();
    for(size_t i=0; i < arr.getSize();i++){
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
    size_t N = 2;
  ComplexMatrix a = createMat(N,N);
  
  ComplexVector lam(N);
  ComplexMatrix W(N,N);

  auto xa = wrap_xmux(a);
  auto xl = wrap_xmux(lam);
  auto xw = wrap_xmux(W);

  eig_gpu(xa,xl,xw);
  xl.to_cpu();
  xw.to_cpu();

  show_arr(lam);
  show_arr(W);

  return true;
}
