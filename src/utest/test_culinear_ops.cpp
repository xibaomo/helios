#include "arr.h"
#include "cuhandlemgr.h"
#include "utest.h"
#include "xmux.h"
using namespace std;

static bool test_transpose();
static AddUnitTest t_trans("test_transpose", test_transpose);

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
