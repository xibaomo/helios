#include "xrcwa2d.h"

#include "culinear.h"
#include "xmux.h"

SMat XRcwa2D::createLayerSmat(const ComplexMatrix& W, const ComplexMatrix& V,
                              const ComplexMatrix& Lambda, Real thickness) {
  ComplexMatrix X(W.getSize1(), W.getSize2());
  Real k0 = 2 * Pi / m_lambda;
  for (size_t i = 0; i < X.getSize1(); i++) {
    Complex tmp = -Lambda[i][i] * k0 * thickness;
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

  SMat sm = createLayerSmat(W, V, Lambda, thickness);
  m_scatterMatrices.emplace_back(std::move(sm));
}