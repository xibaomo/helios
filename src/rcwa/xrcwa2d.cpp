#include "xrcwa2d.h"

#include "culinear.h"
#include "xmux.h"

XRcwa2D::XRcwa2D(Real lambda, Real Lx, Real Ly, size_t max_order_x, size_t max_order_y,
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

void XRcwa2D::createKMatrices() {
  // generate k grids (-N,-N), (-N, -N+1) ... (0,0), (0,1) ... (N,N)
  m_kgrids.resize(m_orderN);
  size_t k = 0;
  for (size_t i = -m_maxOrderXY[0]; i <= m_maxOrderXY[0]; i++) {
    for (size_t j = -m_maxOrderXY[1]; j <= m_maxOrderXY[1]; j++) {
      m_kgrids[k++] = {i, j};
    }
  }
  // fill Kx: kx_inc - m*2pi/Lx, unit of k0
  m_Kx_norm.resize(m_orderN, m_orderN);
  m_Kx_norm.zero();
  m_Ky_norm = m_Kx_norm;

  for (size_t i = 0; i < m_orderN; i++) {
    m_Kx_norm[i][i] = m_kx_inc_norm - m_kgrids[i].first * m_lambda / m_Lx;
    m_Ky_norm[i][i] = m_ky_inc_norm - m_kgrids[i].second * m_lambda / m_Ly;
  }
}

SMat XRcwa2D::createLayerSmat(const ComplexMatrix& W, const ComplexMatrix& V,
                              const ComplexMatrix& Lambda, Real thickness) {
  ComplexMatrix X(W.getSize1(), W.getSize2());
  for (size_t i = 0; i < X.getSize1(); i++) {
    Complex tmp = -Lambda[i][i] * m_k0 * thickness;
    X[i][i] = std::exp(tmp);
  }

  auto xW0 = wrap_xmux(m_W0);
  auto xV0 = wrap_xmux(m_V0);
  auto xW = wrap_xmux(W);
  auto xV = wrap_xmux(V);
  auto xL = wrap_xmux(Lambda);
  auto xX = wrap_xmux(X);

  XMux<ComplexMatrix> A1(W.getSize1(), W.getSize2());
  XMux<ComplexMatrix> A2(W.getSize1(), W.getSize2());
  linsolve_mat_gpu(xW, xW0, A1);
  linsolve_mat_gpu(xV, xV0, A2);

  A1.add(A2);

  XMux<ComplexMatrix> B1(W.getSize1(), W.getSize2());
  XMux<ComplexMatrix> B2 = B1;
  linsolve_mat_gpu(xW, xW0, B1);
  linsolve_mat_gpu(xV, xV0, B2);
  B1.sub(B2);

  // compute U = (A-XBA^-1XB)
  // compute K = A^-1 X
  XMux<ComplexMatrix> K(W.getSize1(), W.getSize2());
  XMux<ComplexMatrix> U = A1;
  linsolve_mat_gpu(A1, xX, K);
  XMux<ComplexMatrix> U1 = xX * B1 * K * B1;
  U.sub(U1);
  // compute G=XBA^-1XA-B
  XMux<ComplexMatrix> G = xX * B1 * K * A1;
  G.sub(B1);

  SMat sm;
  linsolve_mat_gpu(U, G, sm.s11);

  XMux<ComplexMatrix> C;
  linsolve_mat_gpu(A1, B1, C);  // C=A^-1 B
  C = B1 * C;
  XMux<ComplexMatrix> C1 = A1;
  C1.sub(C);

  XMux<ComplexMatrix> C2;
  linsolve_mat_gpu(U, xX, C2);
  sm.s12 = C2 * C1;

  return sm;
}

void XRcwa2D::addUniformLayer(const Complex& eps, Real thickness) {
  m_thicknessList.push_back(thickness);

  ComplexMatrix Kz(m_orderN, m_orderN);
  for (size_t i = 0; i < m_orderN; i++) {
    auto tmp = std::conj(eps) - m_Kx_norm[i][i] * m_Kx_norm[i][i] -
               m_Ky_norm[i][i] * m_Ky_norm[i][i];
    Kz[i][i] = std::conj(std::sqrt(tmp));
  }

  ComplexMatrix W(m_orderN * 2, m_orderN * 2);
  ComplexMatrix Lambda = W;
  ComplexMatrix V = W;
  ComplexMatrix Q = W;
  W.eye();

  // fill Lambda
  for (size_t i = 0; i < m_orderN; i++) {
    Lambda[i][i] = JJ * Kz[i][i];
    Lambda[i + m_orderN][i + m_orderN] = JJ * Kz[i][i];
  }
  // fill Q
  for (size_t i = 0; i < m_orderN; i++) {
    // diagonal blocks
    Q[i][i] = m_Kx_norm[i][i] * m_Ky_norm[i][i];
    Q[i + m_orderN][i + m_orderN] = -m_Kx_norm[i][i] * m_Ky_norm[i][i];

    // UR block
    Q[i][i + m_orderN] = eps - m_Kx_norm[i][i] * m_Kx_norm[i][i];
    // LL block
    Q[i + m_orderN][i] = m_Ky_norm[i][i] * m_Ky_norm[i][i] - eps;
  }

  auto xV = wrap_xmux(V);
  auto xQ = wrap_xmux(Q);
  auto xL = wrap_xmux(Lambda);

  linsolve_right_gpu(xL, xQ, xV);
  xV.to_cpu();

  if (eps == COMPLEX_ONE) {
    m_W0 = W;
    m_V0 = V;
  }

  SMat sm = createLayerSmat(W, V, Lambda, thickness);
  m_scatterMatrices.emplace_back(std::move(sm));
}

ComplexMatrix XRcwa2D::createConvMat(const Array2D<Complex>& eps_img, int max_order) {
  size_t nx = eps_img.getSize1();
  size_t ny = eps_img.getSize2();
  ComplexMatrix F_eps_raw = eps_img;
  auto mx_F_eps = wrap_xmux(F_eps_raw);
  fft2d(mx_F_eps);
  mx_F_eps.scale(1.f/(nx*ny));
  ComplexMatrix eps_mat(m_orderN, m_orderN);

  return eps_mat;
}

void XRcwa2D::addPatternLayer(const Array2D<Complex>& eps, Real thickness) {}