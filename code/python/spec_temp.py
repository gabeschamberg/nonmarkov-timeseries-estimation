#####
#
# Applies the ADMM code to the problem of spectrotemporal decompositions
#
#####
#
# lrsd - Returns len(signal)//L sets of K frequency coefficients for the time
#       series passed in (signal) based on the low-rank spectrotemporal
#       decomposition algorithm
#
# spec_pursuit - Returns len(signal)//L sets of K frequency coefficients for the
#       time series passed in (signal) based on the spectrotemporal pursuit
#       algorithm
#
#####

import numpy as np
from matplotlib import mlab
from framework import AdmmObject, AdmmSolver
from updates import fourier_x, matrix_z, group_sparse_w, nuc_norm_w
from updates import matrix_lam, matrix_alph
from helpers import fourier_mat, est_noise


# Creates a spectrotemporal decomposition where the matrix of differences
# between columns is LOW-RANK
def lrsd(signal, rho, K, L, beta, overlap=0,
         handle_artifacts=False, noise_range=(1, 5), noise_vars=None,
         thresh=0.0001, max_iters=10, verbosity=0):
    return spectemp_decomposition(nuc_norm_w,
                                  signal, rho, K, L, beta, overlap,
                                  handle_artifacts, noise_range, noise_vars,
                                  thresh, max_iters, verbosity)


# Creates a spectrotemporal decomposition where the matrix of differences
# between columns is GROUP-SPARSE
def spec_pursuit(signal, rho, K, L, beta, overlap=0,
                 handle_artifacts=False, noise_range=(1, 5), noise_vars=None,
                 thresh=0.0001, max_iters=10, verbosity=0):
    return spectemp_decomposition(group_sparse_w,
                                  signal, rho, K, L, beta, overlap,
                                  handle_artifacts, noise_range, noise_vars,
                                  thresh, max_iters, verbosity)


# Conducts all the setup for BOTH spectrotemporal decomposition methods
# where the only difference is the w_update parameter
def spectemp_decomposition(w_update, signal, rho, K, L, beta, overlap,
                           handle_artifacts, noise_range, noise_vars, thresh, max_iters, verbosity):
    N = int(((len(signal) // L) - 1) * (1 / (1 - overlap)))
    admmobj = AdmmObject(N, signal, rho,
                         fourier_x, w_update, matrix_z,
                         matrix_lam, matrix_alph)
    admmobj.K = K
    admmobj.L = L
    admmobj.beta = beta
    admmobj.overlap = overlap
    admmobj.handle_artifacts = handle_artifacts
    admmobj.x = np.zeros((K, N))
    admmobj.w = np.zeros((K, N))
    admmobj.z = np.zeros_like(admmobj.w)
    admmobj.lam = np.random.randn(K, N)
    admmobj.alph = np.random.randn(K, N)
    # If artifact noise is a concern, will estimate noise variance per window
    if handle_artifacts:
        if verbosity > 0:
            print('Handling Artifacts - Creating Cinv for each n')
        if noise_vars is not None:
            admmobj.sigsq = noise_vars
        else:
            admmobj.sigsq = est_noise(signal, K, N, noise_range)
    else:
        admmobj.sigsq = np.ones(N)
    # If L is not a multiple of K then we need many F matrices
    if np.mod(L, K) != 0:
        if verbosity > 0:
            print('L%K != 0 - Creating F for each n')
        admmobj.oneF = False
        admmobj.F = []
        admmobj.Cinv = []
        for n in range(admmobj.N):
            admmobj.F.append(fourier_mat(L, K, n * L * (1 - overlap)))
            admmobj.Cinv.append(np.linalg.inv((1 / admmobj.sigsq[n]) * \
                                              admmobj.F[n].T.dot(admmobj.F[n]) + rho * np.eye(K)))
    elif handle_artifacts:
        admmobj.oneF = True
        admmobj.F = fourier_mat(L, K)
        admmobj.Cinv = []
        for n in range(admmobj.N):
            admmobj.Cinv.append(np.linalg.inv((1 / admmobj.sigsq[n]) * \
                                              admmobj.F.T.dot(admmobj.F) + rho * np.eye(K)))
    else:
        admmobj.oneF = True
        admmobj.F = fourier_mat(L, K)
        admmobj.Cinv = np.linalg.inv((1 / admmobj.sigsq[0]) * \
                                     admmobj.F.T.dot(admmobj.F) + rho * np.eye(K))

    admmslv = AdmmSolver(admmobj, thresh, max_iters, verbosity)
    (x, w, z, lam, alpha) = admmslv.solve()
    if verbosity > 0:
        admmslv.print_results()
    if verbosity > 1:
        admmslv.plot_residuals()
    x_cmplx = x[0:K // 2, :] - x[K // 2:K, :] * 1j
    x_mag = 10 * np.log10(np.abs(x_cmplx) ** 2)
    return x, x_mag
