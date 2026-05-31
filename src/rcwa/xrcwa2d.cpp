#include "xrcwa2d.h"

#include "culinear.h"
#include "utils.h"
#include "xmux.h"

XRcwa2D::XRcwa2D(Real lambda, Real Lx, Real Ly, size_t max_order_x,
                 size_t max_order_y, Real theta, Real phi, Complex& in_eps)
    : m_lambda(lambda), m_theta(theta), m_phi(phi), m_eps_ref(in_eps) {
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

  // vacuum properties
  prepareVacuum();
  // initialize global s matrix
  m_global_smat.resize(m_orderN * 2, m_orderN * 2);
  m_global_smat.s12.eye();
  m_global_smat.s21.eye();
}

void XRcwa2D::prepareVacuum() {
  auto xKx = wrap_xmux(m_Kx_norm);
  auto xKy = wrap_xmux(m_Ky_norm);
  XMux<ComplexMatrix> I0 = m_Kx_norm;
  I0.eye();
  m_xW0.resize(m_orderN * 2, m_orderN * 2);
  m_xW0.eye();
  XMux<ComplexMatrix> lam0(m_orderN * 2, m_orderN * 2);
  lam0.zero();
  XMux<ComplexMatrix> tmp = m_Kz0_norm;
  tmp.scale(Complex{0.f, 1.f});
  lam0.fillBlock(0, 0, tmp);
  lam0.fillBlock(m_orderN, m_orderN, tmp);
  XMux<ComplexMatrix> Q0;
  Q0.fillBlock(0, 0, xKx * xKy);
  auto Q12 = xKx;
  Q12.eye();
  Q12.substract(xKx * xKx);
  Q0.fillBlock(0, m_orderN, Q12);

  auto Q21 = xKy * xKy;
  Q21.substract(I0);
  Q0.fillBlock(m_orderN, 0, Q21);

  auto Q22 = xKy;
  Q22.zero();
  Q22.substract(xKx * xKy);
  Q0.fillBlock(m_orderN, m_orderN, Q22);

  linsolve_right_gpu(lam0, Q0, m_xV0);
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

    m_Kz_norm_ref[i][i] = -std::conj(
        std::sqrt(std::conj(m_eps_ref) - m_Kx_norm[i][i] * m_Kx_norm[i][i] -
                  m_Ky_norm[i][i] * m_Ky_norm[i][i]));
    m_Kz_norm_trn[i][i] = std::conj(
        std::sqrt(std::conj(m_eps_trn) - m_Kx_norm[i][i] * m_Kx_norm[i][i] -
                  m_Ky_norm[i][i] * m_Ky_norm[i][i]));
    m_Kz0_norm[i][i] = std::conj(std::sqrt(Complex{1.f, 0.f} -
                                           m_Kx_norm[i][i] * m_Kx_norm[i][i] -
                                           m_Ky_norm[i][i] * m_Ky_norm[i][i]));
  }
}

SMat XRcwa2D::createLayerSmat(const ComplexMatrix& W, const ComplexMatrix& V,
                              const ComplexMatrix& Lambda, Real thickness) {
  ComplexMatrix X(W.getSize1(), W.getSize2());
  for (size_t i = 0; i < X.getSize1(); i++) {
    Complex tmp = -Lambda[i][i] * m_k0 * thickness;
    X[i][i] = std::exp(tmp);
  }

  auto xW = wrap_xmux(W);
  auto xV = wrap_xmux(V);
  auto xL = wrap_xmux(Lambda);
  auto xX = wrap_xmux(X);

  XMux<ComplexMatrix> A1(W.getSize1(), W.getSize2());
  XMux<ComplexMatrix> A2(W.getSize1(), W.getSize2());
  linsolve_mat_gpu(xW, m_xW0, A1);
  linsolve_mat_gpu(xV, m_xV0, A2);

  A1.add(A2);

  XMux<ComplexMatrix> B1(W.getSize1(), W.getSize2());
  XMux<ComplexMatrix> B2 = B1;
  linsolve_mat_gpu(xW, m_xW0, B1);
  linsolve_mat_gpu(xV, m_xV0, B2);
  B1.substract(B2);

  // compute U = (A-XBA^-1XB)
  // compute K = A^-1 X
  XMux<ComplexMatrix> K(W.getSize1(), W.getSize2());
  XMux<ComplexMatrix> U = A1;
  linsolve_mat_gpu(A1, xX, K);
  XMux<ComplexMatrix> U1 = xX * B1 * K * B1;
  U.substract(U1);
  // compute G=XBA^-1XA-B
  XMux<ComplexMatrix> G = xX * B1 * K * A1;
  G.substract(B1);

  SMat sm;
  linsolve_mat_gpu(U, G, sm.s11);

  XMux<ComplexMatrix> C;
  linsolve_mat_gpu(A1, B1, C);  // C=A^-1 B
  C = B1 * C;
  XMux<ComplexMatrix> C1 = A1;
  C1.substract(C);

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

  SMat sm = createLayerSmat(W, V, Lambda, thickness);
  // update global s matrix
  m_global_smat = redheffer(m_global_smat, sm);
}

void XRcwa2D::buildSMat_reflection() {
  // reflection region
  if (m_eps_ref == Complex{1.f, 0.f}) return;
  auto mx_Kx = wrap_xmux(m_Kx_norm);
  auto mx_Ky = wrap_xmux(m_Ky_norm);
  // Qref=[Kx*Ky,eps_ref*I0-Kx*Kx;Ky*Ky-eps_ref*I0,-Ky*Kx];
  auto Q11 = mx_Kx * mx_Ky;
  XMux<ComplexMatrix> eps_I0 = Q11;
  eps_I0.eye();
  eps_I0.scale(m_eps_ref);
  XMux<ComplexMatrix> Z0 = Q11;
  Z0.zero();

  XMux<ComplexMatrix> Q12 = eps_I0;
  Q12.substract(mx_Kx * mx_Kx);

  XMux<ComplexMatrix> Q21 = mx_Ky * mx_Ky;
  Q21.substract(eps_I0);

  XMux<ComplexMatrix> Q22 = Z0;
  Q22.substract(mx_Ky * mx_Kx);

  // Qref=[Kx*Ky,eps_ref*I0-Kx*Kx;Ky*Ky-eps_ref*I0,-Ky*Kx];
  XMux<ComplexMatrix> Qref(m_orderN * 2, m_orderN * 2);
  Qref.fillBlock(0, 0, Q11);
  Qref.fillBlock(0, m_orderN, Q12);
  Qref.fillBlock(m_orderN, 0, Q21);
  Qref.fillBlock(m_orderN, m_orderN, Q22);

  // Wref=eye(2*orderN,2*orderN);
  // Lam_ref = [-1i*Kz_ref,Z0;Z0, -1i*Kz_ref];
  // Vref = Qref/(Lam_ref);
  // A = W0\Wref + V0\Vref;
  // B = W0\Wref - V0\Vref;
  // Sr_11 = -A\B;
  // Sr_12 = 2*inv(A);
  // Sr_21 = 0.5*(A-B/A*B);
  // Sr_22 = B/A;
  // [S11,S12,S21,S22]=redheffer(Sr_11,Sr_12,Sr_21,Sr_22,S11,S12,S21,S22);

  XMux<ComplexMatrix> Lam_ref = Qref;
  XMux<ComplexMatrix> tmp = m_Kz_norm_ref;
  tmp.scale(Complex{0.f, -1.f});
  Lam_ref.fillBlock(0, 0, tmp);
  Lam_ref.fillBlock(m_orderN, m_orderN, tmp);
  XMux<ComplexMatrix> Vref;
  linsolve_right_gpu(Lam_ref, Qref, Vref);

  XMux<ComplexMatrix> A(m_orderN * 2, m_orderN * 2);
  A.eye();
  XMux<ComplexMatrix> B = A;

  // tmp = V0^-1 * Vref
  linsolve_mat_gpu(m_xV0, Vref, tmp);
  A.add(tmp);
  B.substract(tmp);

  SMat s_ref;
  s_ref.resize(m_orderN * 2, m_orderN * 2);
  linsolve_mat_gpu(A, B, s_ref.s11);
  s_ref.s11.scale(-1.f);

  XMux<ComplexMatrix> I0 = A;
  I0.eye();
  linsolve_mat_gpu(A, I0, s_ref.s12);
  s_ref.s12.scale(2.f);

  tmp = A;
  // tmp = B*A^-1
  linsolve_right_gpu(A, B, tmp);
  s_ref.s22 = tmp;
  tmp = tmp * B;
  s_ref.s21 = A;
  s_ref.s21.substract(tmp);
  s_ref.s21.scale(0.5f);

  m_global_smat = redheffer(s_ref, m_global_smat);
}
void XRcwa2D::buildSMat_transmission() {
  if (m_eps_trn == Complex{1.f, 0.f}) return;
  auto mx_Kx = wrap_xmux(m_Kx_norm);
  auto mx_Ky = wrap_xmux(m_Ky_norm);
  // Qtrn=[Kx*Ky,eps_ref*I0-Kx*Kx;Ky*Ky-eps_ref*I0,-Ky*Kx];
  auto Q11 = mx_Kx * mx_Ky;
  XMux<ComplexMatrix> eps_I0 = Q11;
  eps_I0.eye();
  eps_I0.scale(m_eps_trn);
  XMux<ComplexMatrix> Z0 = Q11;
  Z0.zero();

  XMux<ComplexMatrix> Q12 = eps_I0;
  Q12.substract(mx_Kx * mx_Kx);

  XMux<ComplexMatrix> Q21 = mx_Ky * mx_Ky;
  Q21.substract(eps_I0);

  XMux<ComplexMatrix> Q22 = Z0;
  Q22.substract(mx_Ky * mx_Kx);

  // Qtrn=[Kx*Ky,eps_ref*I0-Kx*Kx;Ky*Ky-eps_ref*I0,-Ky*Kx];
  XMux<ComplexMatrix> Qtrn(m_orderN * 2, m_orderN * 2);
  Qtrn.fillBlock(0, 0, Q11);
  Qtrn.fillBlock(0, m_orderN, Q12);
  Qtrn.fillBlock(m_orderN, 0, Q21);
  Qtrn.fillBlock(m_orderN, m_orderN, Q22);

  XMux<ComplexMatrix> Lam_trn = Qtrn;
  Lam_trn.zero();
  XMux<ComplexMatrix> tmp = m_Kz_norm_trn;
  tmp.scale(Complex{0.f, 1.f});
  Lam_trn.fillBlock(0, 0, tmp);
  Lam_trn.fillBlock(m_orderN, m_orderN, tmp);
  XMux<ComplexMatrix> Vtrn;
  linsolve_right_gpu(Lam_trn, Qtrn, Vtrn);

  XMux<ComplexMatrix> A(m_orderN * 2, m_orderN * 2);
  A.eye();
  XMux<ComplexMatrix> B = A;

  // tmp = V0^-1 * Vtrn
  linsolve_mat_gpu(m_xV0, Vtrn, tmp);
  A.add(tmp);
  B.substract(tmp);

  SMat s_trn;
  linsolve_right_gpu(A, B, s_trn.s11);
  tmp = s_trn.s11 * B;
  s_trn.s12 = A;
  s_trn.s12.substract(tmp);
  s_trn.s12.scale(0.5f);

  XMux<ComplexMatrix> I0 = A;
  I0.eye();
  linsolve_mat_gpu(A, I0, s_trn.s21);
  s_trn.s21.scale(2.f);

  linsolve_mat_gpu(A, B, s_trn.s22);
  s_trn.s22.scale(-1.f);

  m_global_smat = redheffer(m_global_smat, s_trn);
}

void XRcwa2D::buildGlobalSMat() {
  buildSMat_reflection();
  buildSMat_transmission();
}

void XRcwa2D::addPatternLayer(const Array2D<Complex>& eps, Real thickness) {}

SMat XRcwa2D::redheffer(SMat& a, SMat& b) {
  SMat res;
  XMux<ComplexMatrix> I0 = a.s11;
  I0.eye();

  XMux<ComplexMatrix> D = I0;
  XMux<ComplexMatrix> F = I0;
  D.substract(b.s11 * a.s22);
  F.substract(a.s22 * b.s11);
  linsolve_right_inplace_gpu(D, a.s12);
  linsolve_right_inplace_gpu(F, b.s21);
  D = a.s12;
  F = b.s21;

  res.s11 = a.s11;
  res.s11.add(D * b.s11 * a.s21);

  res.s12 = D * b.s12;
  res.s21 = F * a.s21;
  res.s22 = b.s22;
  res.s22.add(F * a.s22 * b.s12);

  return res;
}