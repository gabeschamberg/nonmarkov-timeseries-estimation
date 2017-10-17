#####
#
# Contains various functions that solve the X, Z, and W updates. These functions
# would be passed to AdmmObjects so that they can be used in iterative updates.
#
#####

import numpy as np
from helpers import column_diffs, bt_solver


#################### X Updates #####################
# x update function used for EEG model
def fourier_x(admmobj):
    # problem is completely separable in n
    for n in range(admmobj.N):
        start = int(n * (admmobj.L * (1 - admmobj.overlap)))
        y = admmobj.obs[start:start + admmobj.L]
        lam = admmobj.lam[:, n]
        z = admmobj.z[:, n]
        if admmobj.oneF:
            b = ((1 / admmobj.sigsq[n]) * np.dot(admmobj.F.T, y) - lam +
                  admmobj.rho * z)
        else:
            b = ((1 / admmobj.sigsq[n]) * np.dot(admmobj.F[n].T, y) - lam +
                  admmobj.rho * z)
        if admmobj.handle_artifacts or (not admmobj.oneF):
            admmobj.x[:, n] = admmobj.Cinv[n].dot(b)
        else:
            admmobj.x[:, n] = admmobj.Cinv.dot(b)


# x update when observations are modeled as independent linear gaussian measurement
def gaussian_x(admmobj):
    if admmobj.K > 1:
        for n in range(admmobj.N):
            admmobj.x[:, n] = 0.5 * admmobj.Cinv.dot((1 / admmobj.sig) * \
                np.dot(admmobj.A.T, admmobj.obs[:, n]) + \
                (admmobj.rho * admmobj.z[:, n] - admmobj.lam[:, n]))
    else:
        for n in range(admmobj.N):
            admmobj.x[n] = ((admmobj.obs[n] - admmobj.b) * admmobj.A / \
                admmobj.sig + admmobj.z[n] * admmobj.rho / 2 - 0.5 * \
                admmobj.lam[n]) / (admmobj.A ** 2 / admmobj.sig + \
                admmobj.rho / 2)

def ssml_solve_n_cvx(admmobj, n, const1):
    import cvxpy as cvx
    M = admmobj.obs[0]
    Q = admmobj.obs[1]
    R = admmobj.obs[2]
    R_n = R[n, :]
    const2 = -admmobj.g * sum(R_n)
    x = cvx.Variable(2)
    constraints = [x[1] == -admmobj.mu / admmobj.eta]
    obj = cvx.Minimize(
        ((admmobj.h ** 2) / (2 * admmobj.sigg) + admmobj.rho / 2) *
        cvx.square(x[0]) +
        (admmobj.lam[n] - M[n] * admmobj.eta -
         ((Q[n] - admmobj.delta) * admmobj.h) / admmobj.sigg -
         admmobj.rho * admmobj.z[n]) *
        x[0] + cvx.log_sum_exp(admmobj.mu + admmobj.eta * x) +
        admmobj.Del * cvx.exp(admmobj.psi + admmobj.g * x[0]) * const1 +
        const2 * x[0]
    )
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.SCS)
    return x.value[0, 0]

def ssml_solve_n(admmobj, n, const1):
    import cvxpy as cvx
    M = admmobj.obs[0]
    Q = admmobj.obs[1]
    R = admmobj.obs[2]
    R_n = R[n, :]
    x = admmobj.x[n]
    eps = 10**(-5)
    diff = eps + 1
    while(diff > eps):

        df_M = admmobj.eta*np.exp(admmobj.mu+admmobj.eta*x)/ \
            (1+np.exp(admmobj.mu+admmobj.eta*x)) - \
            M[n]*admmobj.eta
        df_Q = -admmobj.h*\
            (Q[n] - admmobj.delta - admmobj.h*x)/ \
            (admmobj.sigg)
        df_R = admmobj.g*admmobj.Del*\
            np.exp(admmobj.psi + admmobj.g*x)*const1 - \
            admmobj.g*sum(R_n)
        df_prox = admmobj.rho * (x - admmobj.z[n] + \
            admmobj.lam[n]/admmobj.rho)
        df = df_M + df_Q + df_R + df_prox

        ddf_M = (admmobj.eta**2) *\
            np.exp(admmobj.mu+admmobj.eta*x)/ \
            ((1+np.exp(admmobj.mu+admmobj.eta*x))**2)
        ddf_Q = (admmobj.h**2) / (admmobj.sigg)
        ddf_R = (admmobj.g**2)*admmobj.Del*\
            np.exp(admmobj.psi + admmobj.g*x)*const1
        ddf_prox = admmobj.rho
        ddf = ddf_M + ddf_Q + ddf_R + ddf_prox

        x = x - df/ddf
        diff = np.abs(df/ddf)
    return x


def ssml_grad_step(admmobj, n):
    step = 0.01
    num_steps = 1
    M = admmobj.obs[0]
    Q = admmobj.obs[1]
    R = admmobj.obs[2]
    S = len(admmobj.c)
    x = admmobj.x[n]
    c1 = admmobj.h ** 2 / (2 * admmobj.sigg) + admmobj.rho / 2
    c2 = admmobj.lam[n] - M[n] * admmobj.eta - admmobj.h * \
    (Q[n] - admmobj.delta) / admmobj.sigg - admmobj.rho * admmobj.z[n]
    c3 = 0
    for j in range(admmobj.J):
        temp = 0
        for s in range(S):
            if j - (s + 1) >= 0:
                temp += admmobj.c[s] * R[n][j - (s + 1)]
        c3 += np.exp(temp)
    c4 = -admmobj.g * sum(R[n, :])
    for i in range(num_steps):
        grad = 2 * c1 * x + c2 + admmobj.eta * np.exp(admmobj.mu + admmobj.eta * x) / (
        1 + np.exp(admmobj.mu + admmobj.eta * x)) + \
               admmobj.Del * c3 * admmobj.g * np.exp(admmobj.psi + admmobj.g * x) + c4
        x -= step * grad
    return x


def ssml_x(admmobj):
    c3 = np.zeros_like(admmobj.x)
    R = admmobj.obs[2]
    S = len(admmobj.c)
    for j in range(admmobj.J):
        c3 += np.exp(np.dot(R[:, max(0, j - S + 1):j + 1],
            np.asarray(admmobj.c[max(S - j - 1, 0):S]).T))
    for n in range(admmobj.N):
        admmobj.x[n] = ssml_solve_n(admmobj, n, c3[n])


def ssml_x_grad(admmobj):
    for n in range(admmobj.N):
        admmobj.x[n] = ssml_grad_step(admmobj, n)


def ssml_x_parallel(admmobj):
    from joblib import Parallel, delayed
    import multiprocessing
    c3 = np.zeros_like(admmobj.x)
    R = admmobj.obs[2]
    S = len(admmobj.c)
    for j in range(admmobj.J):
        c3 += np.exp(np.dot(R[:, max(0, j - S + 1):j + 1], np.asarray(admmobj.c[max(S - j - 1, 0):S]).T))
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(ssml_solve_n)(admmobj, n, c3[n]) for n in range(admmobj.N))
    admmobj.x = np.asarray(results)


def ssml_x_grad_parallel(admmobj):
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(ssml_grad_step)(admmobj, n) for n in range(admmobj.N))
    admmobj.x = np.asarray(results)


#################### Z Updates #####################
def matrix_z(admmobj):
    zx = admmobj.x + (admmobj.lam / admmobj.rho)
    zw = admmobj.w + (admmobj.alph / admmobj.rho)
    veczx = np.zeros((admmobj.N * admmobj.K, 1))
    veczw = np.zeros((admmobj.N * admmobj.K, 1))
    for n in range(admmobj.N):
        veczx[n * admmobj.K:(n + 1) * admmobj.K] = zx[:, n].reshape((admmobj.K, 1))
        veczw[n * admmobj.K:(n + 1) * admmobj.K] = zw[:, n].reshape((admmobj.K, 1))
    veczw[0:(admmobj.N - 1) * admmobj.K] = veczw[0:(admmobj.N - 1) * admmobj.K] - \
                                           veczw[admmobj.K:admmobj.N * admmobj.K]
    vecz = bt_solver(veczx + veczw, admmobj.N, admmobj.K)
    for n in range(admmobj.N):
        admmobj.z[:, n] = vecz[n * admmobj.K:(n + 1) * admmobj.K].reshape((admmobj.K))


def vector_z(admmobj):
    # a and b simplify the results of completing the square
    a = admmobj.x + admmobj.lam / admmobj.rho
    b = admmobj.w + admmobj.alph / admmobj.rho
    z = np.dot(admmobj.igginv, (a + np.dot(admmobj.G.transpose(), b)))
    admmobj.z = z


#################### W Updates #####################
# w update used for spectral sparsity and temporal smoothness
def group_sparse_w(admmobj):
    # Create matrix that is z differences
    Z = column_diffs(admmobj.z)
    A = Z - (admmobj.alph / admmobj.rho)
    w = np.zeros(admmobj.w.shape)
    for k in range(admmobj.K):
        Ak = A[k, :]
        Ak_norm = np.linalg.norm(Ak)
        w[k, :] = max(0, ((Ak_norm - (admmobj.beta / admmobj.rho)) / Ak_norm)) * Ak
    admmobj.w = w

# w update used to enforce low-rank in w
def nuc_norm_w(admmobj):
    # Create matrix that is z differences
    Z = column_diffs(admmobj.z)
    A = Z - admmobj.alph / admmobj.rho
    v = admmobj.beta / admmobj.rho
    U, s, V = np.linalg.svd(A, full_matrices=False)
    sbar = np.empty_like(s)
    for i in range(len(s)):
        if s[i] - v > 0:
            sbar[i] = s[i] - v
        else:
            sbar[i] = 0
    Sbar = np.diag(sbar)
    admmobj.w = U.dot(Sbar).dot(V)


# w update used for gauss-markov latent variable
def least_sq_w(admmobj):
    admmobj.w = (1 / (admmobj.beta / admmobj.sigv + admmobj.rho)) * \
                (admmobj.beta * admmobj.gamma / admmobj.sigv + \
                 admmobj.rho * np.dot(admmobj.G, admmobj.z) - admmobj.alph)

def students_t_w(admmobj):
    eps = 0.0000001
    sig = 0.1
    mu = admmobj.gamma
    rho = admmobj.rho
    w_tild = np.dot(admmobj.G, admmobj.z) - admmobj.alph/admmobj.rho
    for n in range(len(admmobj.w)):
        w = mu
        f_ = eps + 1
        while(np.abs(f_) > eps):
            f_ = 4*(w-mu)/(3*sig + (w-mu)**2) + rho*(w - w_tild[n])
            f__ = (12*sig - 4*((w-mu)**2))/ \
                  ((3*sig + (w-mu)**2)**2) + rho
            w = w - f_/f__
        admmobj.w[n] = w


# w update used to enforce sparsity on a vector w
def l1_w(admmobj):
    Z = np.dot(admmobj.G, admmobj.z)
    A = Z - admmobj.alph / admmobj.rho
    v = admmobj.beta / admmobj.rho
    for n in range(1, admmobj.N):
        if A[n] > 0 and A[n] - v > 0:
            admmobj.w[n] = A[n] - v
        elif A[n] < 0 and A[n] + v < 0:
            admmobj.w[n] = A[n] + v
        else:
            admmobj.w[n] = 0


#################### Lambda Updates #####################
def matrix_lam(admmobj):
    admmobj.lam += admmobj.rho * (admmobj.x - admmobj.z)


def vector_lam(admmobj):
    admmobj.lam += admmobj.rho * (admmobj.x - admmobj.z)


#################### Alpha Updates #####################
def matrix_alph(admmobj):
    Z = column_diffs(admmobj.z)
    admmobj.alph += admmobj.rho * (admmobj.w - Z)


def vector_alph(admmobj):
    admmobj.alph += admmobj.rho * (admmobj.w - np.dot(admmobj.G, admmobj.z))
