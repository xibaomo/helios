#pragma once
#include "arr.h"
#include "physical_constants.h"
#include "types.h"
#include "xmux.h"

struct SMat {
  XMux<ComplexMatrix> s11;  // reflection
  XMux<ComplexMatrix> s12;  // transmission
};

class XRcwa2D {
 private:
  Real m_lambda;
  int m_maxOrderXY[2];
  Real m_Lx,m_Ly;
  Real m_theta, m_phi;  // incident angles

  Array1D<int> m_orderX;
  Array1D<int> m_orderY;
  size_t m_orderN;  // total count of harmonics

  Complex m_eps_in = 1.f;   // in layer epsilon
  Complex m_eps_out = 1.f;  // out layer epsilon

  std::vector<SMat> m_scatterMatrices;  // store s-matrices of all layers
  std::vector<Real> m_thicknessList;

  Complex m_kx_inc_norm = -1;  // unit of k0
  Complex m_ky_inc_norm = -1;  // unit of k0
  Complex m_k0 = -1;

  // vacuum eigen vectors
  ComplexMatrix m_W0;
  ComplexMatrix m_V0;

  // k-matrix
  ComplexMatrix m_Kx_norm;  // kx_inc - m*2pi/Lx, in unit of k0
  ComplexMatrix m_Ky_norm;  // ky_inc - n*2pi/Ly, in unit of k0
  typedef std::pair<int, int> KVector;
  std::vector<KVector> m_kgrids;

 public:
  XRcwa2D(Real lambda, Real Lx, Real Ly, size_t max_order_x, size_t max_order_y,
          Real theta, Real phi, Complex& in_eps)
      : m_lambda(lambda), m_theta(theta), m_phi(phi), m_eps_in(in_eps) {
    m_maxOrderXY[0] = max_order_x;
    m_maxOrderXY[1] = max_order_y;
    m_Lx = Lx;
    m_Ly = Ly;

    m_orderX.resize(2 * max_order_x + 1);
    m_orderY.resize(2 * max_order_y + 1);
    for (size_t i = 0; i < max_order_x; i++) {
      m_orderX[i] = -max_order_x + i;
    }
    for (size_t i = 0; i < max_order_y; i++) {
      m_orderY[i] = -max_order_y + i;
    }
    m_orderN = (2 * m_maxOrderXY[0] + 1) * (2 * m_maxOrderXY[1] + 1);
    std::cout << "Total count of harmonics: " << m_orderN << std::endl;

    m_k0 = 2.f * Pi / m_lambda;
    m_kx_inc_norm = std::sin(theta) * std::cos(phi) * std::sqrt(in_eps);
    m_ky_inc_norm = std::sin(theta) * std::sin(phi) * std::sqrt(in_eps);

    createKMatrices();
  }

  void addUniformLayer(const Complex& eps, Real thickness);
  void addPatternLayer(const Array2D<Complex>& eps, Real thickness);

  void buildGlobalScatterMatrix();

  void createKMatrices();
  
  ComplexMatrix createEpsilonMatrix(const Array2D<Complex>& eps);

  SMat createLayerSmat(const ComplexMatrix& W, const ComplexMatrix& V,
                       const ComplexMatrix& Lambda, Real thickness);
};
