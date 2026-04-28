#include "arr.h"
#include "utest.h"
using namespace std;

static bool test_arr();
static AddUnitTest t_arr("test_arr", test_arr);

static bool test_arr2d();
static AddUnitTest t_arr2d("test_arr2d", test_arr2d);

bool test_arr() {
  size_t N = 4;
  Array1D<Real> a(N);
  for (size_t i = 0; i < N; i++) {
    a[i] = i;
  }

  a.for_each([](Real a) { return a + 1; });

  for (size_t i = 0; i < a.getSize(); i++) {
    cout << a[i] << " ";
  }
  cout << endl;
  return true;
}
bool test_arr2d() {
  size_t N = 2;
  Array2D<Real> a(N, N);
  int k = 0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      a[i][j] = k++;
    }
  }

  a.for_each([](Real a){ return a+1;});

  for (size_t i = 0; i < a.getSize1(); i++) {
    for (size_t j = 0; j < a.getSize2(); j++) {
      cout << a[i][j] << " ";
    }
  }
  cout << endl;
  return true;
}