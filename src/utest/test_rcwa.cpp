#include <iostream>

#include "physical_constants.h"
#include "rcwa/xrcwa2d.h"
#include "utest.h"
#include "utils.h"
using namespace std;

static bool test_rcwa_homogeneous();
static AddUnitTest t_rcwa_homo("test_rcwa_homogeneous", test_rcwa_homogeneous);

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