import numpy as np
from scipy.linalg import expm, eig

# ---------------------------------------------------------------
# helper functions (MATLAB \ and / operators)
# ---------------------------------------------------------------
def mldivide(A, B):
    """A\B  ==  inv(A) @ B"""
    return np.linalg.solve(A, B)

def mrdivide(A, B):
    """A/B  ==  A @ inv(B)"""
    return np.linalg.solve(B.T, A.T).T


# ---------------------------------------------------------------
# redheffer star product
# ---------------------------------------------------------------
def redheffer(a11, a12, a21, a22, b11, b12, b21, b22):
    I0 = np.eye(a11.shape[0], dtype=complex)
    D = a12 @ np.linalg.inv(I0 - b11 @ a22)
    F = b21 @ np.linalg.inv(I0 - a22 @ b11)
    s11 = a11 + D @ b11 @ a21
    s12 = D @ b12
    s21 = F @ a21
    s22 = b22 + F @ a22 @ b12
    return s11, s12, s21, s22


# ---------------------------------------------------------------
# img2conv_mat
# ---------------------------------------------------------------
def img2conv_mat(eps_img, max_order):
    nx, ny = eps_img.shape

    F_eps_raw = np.fft.fft2(eps_img)
    F_eps_coeffs = F_eps_raw / (nx * ny)
    # print(np.argwhere(eps_img==0))
    # breakpoint()

    F_eps_shifted = np.fft.fftshift(F_eps_coeffs)

    kx_fft_range = np.arange(-2 * max_order, 2 * max_order + 1)
    ky_fft_range = np.arange(-2 * max_order, 2 * max_order + 1)

    cx = nx // 2
    cy = ny // 2

    start_kx_idx = cx + kx_fft_range[0]
    end_kx_idx = cx + kx_fft_range[-1]
    start_ky_idx = cy + ky_fft_range[0]
    end_ky_idx = cy + ky_fft_range[-1]

    F_eps_truncated = F_eps_shifted[start_kx_idx:end_kx_idx + 1,
                                     start_ky_idx:end_ky_idx + 1]

    orders_x = 2 * max_order + 1
    orders_y = 2 * max_order + 1
    total_orders = orders_x * orders_y

    eps_conv = np.zeros((total_orders, total_orders), dtype=complex)

    for m_lin_idx in range(total_orders):
        kx_out = m_lin_idx // orders_y - max_order
        ky_out = m_lin_idx % orders_y - max_order
        for p_lin_idx in range(total_orders):
            kx_in = p_lin_idx // orders_y - max_order
            ky_in = p_lin_idx % orders_y - max_order

            kx_diff = kx_out - kx_in
            ky_diff = ky_out - ky_in

            idx_kx_diff = kx_diff - kx_fft_range[0]
            idx_ky_diff = ky_diff - ky_fft_range[0]

            eps_conv[m_lin_idx, p_lin_idx] = F_eps_truncated[idx_kx_diff, idx_ky_diff]

    return eps_conv


# ---------------------------------------------------------------
# generate_normal_field
# ---------------------------------------------------------------
from scipy.ndimage import correlate

def generate_normal_field(epsilon_map_):
    h_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]]) / 8.0
    h_y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]]) / 8.0

    epsilon_map = np.abs(epsilon_map_)

    gx = correlate(epsilon_map, h_x, mode='wrap')
    gy = correlate(epsilon_map, h_y, mode='wrap')

    mag = np.sqrt(gx**2 + gy**2)
    eps_stab = 1e-12

    nx = gx / (mag + eps_stab)
    ny = gy / (mag + eps_stab)

    nx[mag < eps_stab] = 0
    ny[mag < eps_stab] = 0

    return nx, ny


# ---------------------------------------------------------------
# main script
# ---------------------------------------------------------------
def main():
    max_order_x = 1
    max_order_y = 1
    orderN = (2 * max_order_x + 1) * (2 * max_order_y + 1)
    alpha, beta = 0.78, 0
    lam_wave = 193.0     # renamed from 'lambda' (reserved keyword in Python)
    L = 400.0
    n_inc = 1.563
    k0 = 2 * np.pi / lam_wave

    orders_x = np.arange(-max_order_x, max_order_x + 1)
    orders_y = np.arange(-max_order_y, max_order_y + 1)

    # MATLAB meshgrid(orders_x, orders_y) -> shape (len(orders_y), len(orders_x))
    m_grid, n_grid = np.meshgrid(orders_x, orders_y)
    # MATLAB's linear indexing m(i) traverses COLUMN-MAJOR order,
    # so flatten with order='F' to replicate that indexing exactly.
    m_flat = m_grid.flatten(order='F')
    n_flat = n_grid.flatten(order='F')

    sin_theta = 1.35 / 4 * np.sqrt(alpha**2 + beta**2)
    phi = 1 * np.pi / 3
    cos_theta = np.sqrt(1 - sin_theta**2)
    kx_inc = -n_inc * sin_theta * np.cos(phi)
    ky_inc = n_inc * sin_theta * np.sin(phi)
    kz_inc = n_inc * np.sqrt(1 - sin_theta**2)

    Gx = lam_wave / L
    Gy = lam_wave / L

    I0 = np.eye(orderN, dtype=complex)
    Z0 = np.zeros((orderN, orderN), dtype=complex)

    Kx = np.zeros((orderN, orderN), dtype=complex)
    Ky = np.zeros((orderN, orderN), dtype=complex)
    for i in range(orderN):
        Kx[i, i] = kx_inc - m_flat[i] * Gx
        Ky[i, i] = ky_inc - n_flat[i] * Gy

    # vacuum eigen vectors
    W0 = np.eye(orderN * 2, dtype=complex)
    Kz_ref = -np.conj(np.sqrt(n_inc**2 * I0 - Kx @ Kx - Ky @ Ky))
    Kz_trn = np.conj(np.sqrt(I0 - Kx @ Kx - Ky @ Ky))
    Kz0 = Kz_trn

    Q0 = np.block([[Kx @ Ky, I0 - Kx @ Kx],
                    [Ky @ Ky - I0, -Kx @ Ky]])
    lam0 = np.block([[1j * Kz0, Z0],
                      [Z0, 1j * Kz0]])
    V0 = mrdivide(Q0, lam0)

    # global S matrix
    S11 = np.zeros((orderN * 2, orderN * 2), dtype=complex)
    S12 = np.eye(orderN * 2, dtype=complex)
    S21 = S12.copy()
    S22 = S11.copy()

    # homogeneous glass layer
    eps1 = n_inc**2
    t = 200.0
    W1 = np.eye(orderN * 2, dtype=complex)
    Q1 = np.block([[Kx @ Ky, eps1 * I0 - Kx @ Kx],
                    [Ky @ Ky - eps1 * I0, -Ky @ Kx]])
    Kz1 = np.conj(np.sqrt(eps1 * I0 - Kx @ Kx - Ky @ Ky))
    lam1 = np.block([[1j * Kz1, Z0],
                      [Z0, 1j * Kz1]])
    V1 = mrdivide(Q1, lam1)

    A1 = mldivide(W1, W0) + mldivide(V1, V0)
    B1 = mldivide(W1, W0) - mldivide(V1, V0)
    X1 = expm(-lam1 * k0 * t)

    A1_inv = np.linalg.inv(A1)
    D = A1 - X1 @ B1 @ A1_inv @ X1 @ B1
    D_inv = np.linalg.inv(D)

    s1_11 = D_inv @ (X1 @ B1 @ A1_inv @ X1 @ A1 - B1)
    s1_12 = D_inv @ X1 @ (A1 - B1 @ A1_inv @ B1)

    S11, S12, S21, S22 = redheffer(S11, S12, S21, S22,
                                    s1_11, s1_12, s1_12, s1_11)

    # patterned layer
    a = 200.0
    eps = (2.612 - 1j * 0.356) ** 2
    t = 56.0
    s = int(L / 2 - a / 2)   # MATLAB 1-indexed start position

    eps_img = np.ones((int(L), int(L)), dtype=complex)
    # MATLAB: eps_img(s:s+a-1, s:s+a/2-1) = eps   (1-indexed, inclusive)

    eps_img[s:int(s+a), s:s+int(a//2)] = eps

    eps_conv = img2conv_mat(eps_img, max_order_x)
    inv_eps_conv = img2conv_mat(1./eps_img,max_order_x)
    nx, ny = generate_normal_field(eps_img)

    nxx_conv = img2conv_mat(nx * nx, max_order_x)
    nxy_conv = img2conv_mat(nx * ny, max_order_x)
    nyy_conv = img2conv_mat(ny * ny, max_order_x)

    d_eps_conv = inv_eps_conv - eps_conv
    eps_xx_conv = eps_conv + d_eps_conv @ nxx_conv
    eps_xy_conv = d_eps_conv @ nxy_conv
    eps_yx_conv = d_eps_conv @ nxy_conv
    eps_yy_conv = eps_conv + d_eps_conv @ nyy_conv
    
    # breakpoint()

    P11 = Kx @ inv_eps_conv @ Ky
    P12 = I0 - Kx @ inv_eps_conv @ Kx
    P21 = Ky @ inv_eps_conv @ Ky - I0
    P22 = -Ky @ inv_eps_conv @ Kx
    P = np.block([[P11, P12], [P21, P22]])

    Q11 = Kx @ Ky + eps_yx_conv
    Q12 = eps_yy_conv - Kx @ Kx
    Q21 = Ky @ Ky - eps_xx_conv
    Q22 = -Ky @ Kx - eps_xy_conv
    Q = np.block([[Q11, Q12], [Q21, Q22]])

    OMEGA2 = P @ Q

    # eig: scipy.linalg.eig(A) returns (eigvals, eigvecs) with
    # A @ eigvecs[:,i] = eigvals[i] * eigvecs[:,i], matching MATLAB's [W,D]=eig(A)
    eigvals, W = eig(OMEGA2)
    LAM = np.diag(np.sqrt(eigvals))  # principal sqrt, matches MATLAB sqrt() on diagonal

    W_inv = np.linalg.inv(W)
    LAM_inv = np.linalg.inv(LAM)
    V = Q @ W @ LAM_inv

    A = mldivide(W, W0) + mldivide(V, V0)
    B = mldivide(W, W0) - mldivide(V, V0)
    X = expm(-LAM * k0 * t)

    A_inv = np.linalg.inv(A)
    D = A - X @ B @ A_inv @ X @ B
    D_inv = np.linalg.inv(D)

    S2_11 = D_inv @ (X @ B @ A_inv @ X @ A - B)
    S2_12 = D_inv @ X @ (A - B @ A_inv @ B)

    S11, S12, S21, S22 = redheffer(S11, S12, S21, S22,
                                    S2_11, S2_12, S2_12, S2_11)

    # reflection region
    eps_ref = n_inc**2
    Qref = np.block([[Kx @ Ky, eps_ref * I0 - Kx @ Kx],
                      [Ky @ Ky - eps_ref * I0, -Ky @ Kx]])
    Wref = np.eye(2 * orderN, dtype=complex)
    Lam_ref = np.block([[-1j * Kz_ref, Z0],
                         [Z0, -1j * Kz_ref]])
    Vref = mrdivide(Qref, Lam_ref)

    A = mldivide(W0, Wref) + mldivide(V0, Vref)
    B = mldivide(W0, Wref) - mldivide(V0, Vref)

    Sr_11 = -mldivide(A, B)
    Sr_12 = 2 * np.linalg.inv(A)
    Sr_21 = 0.5 * (A - mrdivide(B, A) @ B)
    Sr_22 = mrdivide(B, A)

    S11, S12, S21, S22 = redheffer(Sr_11, Sr_12, Sr_21, Sr_22,
                                    S11, S12, S21, S22)

    # source
    dt = np.zeros((orderN, 1), dtype=complex)
    dt[orderN // 2, 0] = 1.0   # MATLAB: dt(floor(orderN/2)+1)=1  -> 0-indexed same position

    # TE
    px = -np.sin(phi)
    py = -np.cos(phi)
    # TM (uncomment if needed)
    # px = -cos_theta * np.cos(phi)
    # py = cos_theta * np.sin(phi)

    e_src = np.vstack([px * dt, py * dt])
    c_src = e_src
    c_ref = S11 @ c_src
    c_trn = S21 @ c_src
    e_ref = c_ref
    e_trn = c_trn

    rx = e_ref[0:orderN, 0]
    ry = e_ref[orderN:, 0]
    tx = e_trn[0:orderN, 0]
    ty = e_trn[orderN:, 0]

    rz = -np.linalg.inv(Kz_ref) @ (Kx @ rx + Ky @ ry)
    tz = -np.linalg.inv(Kz_trn) @ (Kx @ tx + Ky @ ty)
    
    breakpoint()

    R2 = np.abs(rx)**2 + np.abs(ry)**2 + np.abs(rz)**2
    R = np.real(-Kz_ref).diagonal() / np.real(kz_inc) * R2

    T2 = np.abs(tx)**2 + np.abs(ty)**2 + np.abs(tz)**2
    T = np.real(Kz_trn).diagonal() / np.real(kz_inc) * T2

    mid = len(T) // 2
    print(T[mid])
    print(np.sum(T))


if __name__ == "__main__":
    main()