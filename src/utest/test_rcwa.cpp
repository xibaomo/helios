#include <iostream>

#include "physical_constants.h"
#include "rcwa/xrcwa2d.h"
#include "utest.h"
#include "utils.h"
using namespace std;

static bool test_rcwa_homogeneous();
static AddUnitTest t_rcwa_homo("test_rcwa_homogeneous", test_rcwa_homogeneous);

static bool test_conv_mat();
static AddUnitTest t_conv_mat("test_conv_mat", test_conv_mat);

static bool test_normal_vector_field();
static AddUnitTest t_normal_field("test_normal_vector_field",
                                  test_normal_vector_field);

bool test_rcwa_homogeneous() {
  int max_order_x = 1;
  int max_order_y = 1;

  Real lambda = 193.f;
  Real L = 400.f;  // sim domain size
  Real alpha = 0.78f;
  Real beta = 0.f;  // source point coordinate, in unit of NA
  Real NA = 1.35;
  Real n_inc = 1.563;  // incident medium refractive index: SiO2 glass

  Real k0 = 2 * PI / lambda;
  Real sin_theta = NA / 4 * sqrt(alpha * alpha + beta * beta);  // incident
                                                                // angle
  Real theta = asin(sin_theta);
  Real phi = PI / 3.f;  // azimuthal angle
  Complex eps_in = n_inc * n_inc;
  XRcwa2D rcwa(lambda, L, L, max_order_x, max_order_y, theta, phi, eps_in);
  rcwa.addUniformLayer(eps_in, 200.f);
  rcwa.buildGlobalSMat();

  rcwa.setSourcePolarization(0);

  ComplexVector r = rcwa.getReflectionXY();
  ComplexVector t = rcwa.getTransmissionXY();

  cout << "TE ref: " << endl;
  show_arr(r);
  cout << "TE trn: " << endl;
  show_arr(t);

  cout << "TE power ref: " << endl;
  show_arr(rcwa.getPowerReflectionsAllOrders());

  cout << "TE power trn: " << endl;
  show_arr(rcwa.getPowerTransmissionsAllOrders());

  rcwa.setSourcePolarization(1);

  r = rcwa.getReflectionXY();
  t = rcwa.getTransmissionXY();

  cout << "TM ref: " << endl;
  show_arr(r);
  cout << "TM trn: " << endl;
  show_arr(t);

  cout << "TM power ref: " << endl;
  show_arr(rcwa.getPowerReflectionsAllOrders());

  cout << "TM power trn: " << endl;
  show_arr(rcwa.getPowerTransmissionsAllOrders());

  return true;
}

bool test_conv_mat() {
  int L = 400;
  int a = 80;
  Complex eps = Complex{2.612, -0.356};
  eps = eps * eps;
  ComplexMatrix eps_img(L, L);
  eps_img.for_each([](Complex& a) { return Complex{1.f, 0.f}; });
  int s = L / 2 - a / 2;
  for (int i = s; i < s + a; i++) {
    for (int j = s; j < s + a; j++) {
      eps_img[i][j] = eps;
    }
  }

  ComplexMatrix cm = computeConvMat(eps_img, 1, 1);

  cout << "Eps conv mat: " << endl;
  show_arr(cm);

  auto xm_cm = wrap_xmux(cm);
  cout << "sum of eps_conv" << xm_cm.sum() << endl;

  ComplexMatrix inv_eps = eps_img;
  inv_eps.for_each([](Complex& a) { return 1.f / a; });
  ComplexMatrix inv_cm = computeConvMat(inv_eps, 1, 1);
  auto xm_inv_cm = wrap_xmux(inv_cm);
  cout << "sum of inv eps: " << xm_inv_cm.sum() << endl;
  return true;
}

bool test_normal_vector_field() {
  int L = 400;
  int a = 80;
  Complex eps = Complex{2.612, -0.356};
  eps = eps * eps;
  ComplexMatrix eps_img(L, L);
  eps_img.for_each([](Complex& a) { return Complex{1.f, 0.f}; });
  int s = L / 2 - a / 2;
  for (int i = s; i < s + a; i++) {
    for (int j = s; j < s + a; j++) {
      eps_img[i][j] = eps;
    }
  }

  auto nvf = generateNormalField(eps_img, 1, 1);  // 1nm dx,dy
  auto& [nx, ny] = nvf;

  RealMatrix z = nx;
  z.for_each([](Real a, Real b, Real c) { return b * b + c * c; }, nx, ny);
  cout << "sum of z: " << z.sum() << endl;
  //   cout << "nx: " << endl;
  //   show_arr(z);

  ComplexMatrix nxx = eps_img;
  nxx.for_each([](Complex& a, Real b) { return Complex{b * b, 0.f}; }, nx);
  ComplexMatrix nxx_conv = computeConvMat(nxx, 1, 1);
  cout << "conv nxx sum: " << nxx_conv.sum() << endl;

  ComplexMatrix nxy = eps_img;
  nxy.for_each([](Complex& a, Real b, Real c) { return b * c; }, nx, ny);
  ComplexMatrix nxy_conv = computeConvMat(nxy, 1, 1);
  cout << "conv nxy sum: " << nxy_conv.sum() << endl;

  ComplexMatrix nyy = eps_img;
  nyy.for_each([](Complex& a, Real b) { return Complex{b * b, 0.f}; }, ny);
  ComplexMatrix nyy_conv = computeConvMat(nyy, 1, 1);
  cout << "conv nyy sum: " << nyy_conv.sum() << endl;
  return true;
}