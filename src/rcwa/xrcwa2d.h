#pragma once
#include "arr.h"
#include "physical_constants.h"
#include "types.h"
#include "xmux.h"

struct SMat {
  XMux<ComplexMatrix> s11;  // reflection
  XMux<ComplexMatrix> s12;  // transmission
  XMux<ComplexMatrix> s21;
  XMux<ComplexMatrix> s22;

  void resize(size_t s1, size_t s2) {
    s11.resize(s1,s2); s11.zero();
    s12.resize(s1,s2); s12.zero();
    s21.resize(s1,s2); s21.zero();
    s22.resize(s1,s2); s22.zero();
  }
};  // SMat of a layer is usually symmetric, but overall Smat is not.

class XRcwa2D {
 private:
  Real m_lambda;
  int m_maxOrderXY[2];
  Real m_Lx, m_Ly;
  Real m_theta, m_phi;  // incident angles

  Array1D<int> m_orderX;
  Array1D<int> m_orderY;
  size_t m_orderN;  // total count of harmonics

  Complex m_eps_ref = 1.f;   // in layer epsilon
  Complex m_eps_trn = 1.f;  // out layer epsilon

  std::vector<SMat> m_scatterMatrices;  // store s-matrices of all layers
  std::vector<Real> m_thicknessList;

  Complex m_kx_inc_norm = -1;  // unit of k0
  Complex m_ky_inc_norm = -1;  // unit of k0
  Complex m_k0 = -1;

  // vacuum eigen vectors
  XMux<ComplexMatrix> m_xW0;
  XMux<ComplexMatrix> m_xV0;

  // k-matrix
  ComplexMatrix m_Kx_norm;  // kx_inc - m*2pi/Lx, in unit of k0
  ComplexMatrix m_Ky_norm;  // ky_inc - n*2pi/Ly, in unit of k0
  ComplexMatrix m_Kz_norm_ref;
  ComplexMatrix m_Kz_norm_trn;
  ComplexMatrix m_Kz0_norm; //vacuum Kz
  typedef std::pair<int, int> KVector;
  std::vector<KVector> m_kgrids;

  //global S matrix
  SMat m_global_smat;

  //src array, amplitudes of diffraction orders of Ex,Ey
  ComplexVector m_src;

 public:
  XRcwa2D(Real lambda, Real Lx, Real Ly, size_t max_order_x, size_t max_order_y,
          Real theta, Real phi, Complex& in_eps);

  void addUniformLayer(const Complex& eps, Real thickness);
  void addPatternLayer(const Array2D<Complex>& eps, Real thickness);

  void buildSMat_reflection();
  void buildSMat_transmission();
  void buildGlobalSMat();
  
  void setReflectionRegionEpsilon(Complex& eps) { m_eps_ref = eps;}
  void setTransmissionRegionEpsilon(Complex& eps) { m_eps_trn = eps;}

  void createKMatrices();
  void prepareVacuum();

  SMat createLayerSmat(const ComplexMatrix& W, const ComplexMatrix& V,
                       const ComplexMatrix& Lambda, Real thickness);

  SMat redheffer(SMat& a, SMat& b);

  void setSourcePolarization(int pol = 0); //0: TE, 1: TM

  ComplexVector getReflection();
  ComplexVector getTransmission();
};
