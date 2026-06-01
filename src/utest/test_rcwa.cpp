#include "rcwa/xrcwa2d.h"
#include "utest.h"
#include "physical_constants.h"
#include "utils.h"
#include <iostream>
using namespace std;

static bool test_rcwa_homogeneous();
static AddUnitTest t_rcwa_homo("test_rcwa_homogeneous", test_rcwa_homogeneous);

bool test_rcwa_homogeneous() {
    int max_order = 1;
    int orderN = 2*max_order + 1;
    Real lambda = 193.f;
    Real L = 400.f; //sim domain size 
    Real alpha = 0.78f;
    Real beta = 0.f;   // source point coordinate, in unit of NA
    Real NA = 1.35;
    Real n_inc = 1.563; // incident medium refractive index: SiO2 glass

    Real k0 = 2*PI/lambda;
    Real sin_theta = NA/4*sqrt(alpha*alpha + beta*beta); //incident angle 
    Real theta = asin(sin_theta);
    Real phi = PI/3.f; //azimuthal angle 
    Complex eps_in = n_inc*n_inc;
    XRcwa2D rcwa(lambda,L,L,max_order,max_order,theta,phi,eps_in);
    rcwa.addUniformLayer(eps_in,100.f);
    rcwa.buildGlobalSMat();

    ComplexVector r = rcwa.getReflection();
    ComplexVector t = rcwa.getTransmission();

    cout << "ref: " << endl;
    show_arr(r);
    cout << "trn: " << endl;
    show_arr(t);

    return true;
}