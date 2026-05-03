#include "xrcwa2d.h"

#include "culinear.h"
#include "xmux.h"

SMat XRcwa2D::createLayerSmat(const ComplexMatrix& W, const ComplexMatrix& V,
                              const ComplexMatrix& Lambda, Real thickness) {
  auto xW0 = wrap_xmux(m_W0);
  auto xV0 = wrap_xmux(m_V0);
  auto xW = wrap_xmux(W);
  auto xV = wrap_xmux(V);
  auto xL = wrap_xmux(Lambda);

  XMux<ComplexMatrix> A1(W.getSize1(),W.getSize2());
  XMux<ComplexMatrix> A2(W.getSize1(),W.getSize2());
  linsolve_mat_gpu(xW,xW0,A1);
  linsolve_mat_gpu(xV,xV0,A2);

  return SMat();
}

void XRcwa2D::addUniformLayer(const Complex& eps, Real thickness) {
  m_thicknessList.push_back(thickness);
  ComplexMatrix Kx2(m_orderN, m_orderN);
  ComplexMatrix Ky2(m_orderN, m_orderN);
  ComplexMatrix KxKy(m_orderN, m_orderN);

  for (size_t i = 0; i < m_orderN; i++) {
    Kx2[i][i] = m_kx_inc * m_kx_inc;
    Ky2[i][i] = m_ky_inc * m_ky_inc;
    KxKy[i][i] = m_kx_inc * m_ky_inc;
  }

  size_t mid = m_orderN / 2;
  Kx2[mid][mid] = (m_kx_inc - COMPLEX_ONE) * (m_kx_inc - COMPLEX_ONE);
  Ky2[mid][mid] = (m_ky_inc - COMPLEX_ONE) * (m_ky_inc - COMPLEX_ONE);
  KxKy[mid][mid] = (m_kx_inc - COMPLEX_ONE) * (m_ky_inc - COMPLEX_ONE);

  ComplexMatrix Kz(m_orderN, m_orderN);
  for (size_t i = 0; i < m_orderN; i++) {
    auto tmp = std::conj(eps) - Kx2[i][i] - Ky2[i][i];
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
    Q[i][i] = KxKy[i][i];
    Q[i + m_orderN][i + m_orderN] = -KxKy[i][i];

    // UR block
    Q[i][i + m_orderN] = eps - Kx2[i][i];
    // LL block
    Q[i + m_orderN][i] = Ky2[i][i] - eps;
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
}