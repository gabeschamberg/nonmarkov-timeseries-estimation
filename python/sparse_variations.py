#####
#
# Applies the ADMM code to the problem with a Gaussian measurement model and sparse variations
#
#####
#
# est_x - returns a 1-dimensional vecot estimate of the observed signal with sparse variations
#
#####

import numpy as np
from framework import AdmmObject, AdmmSolver
from updates import gaussian_x, vector_z, l1_w
from updates import vector_lam, vector_alph


# Creates a spectrotemporal decomposition where the matrix of differences
# between columns is LOW-RANK
def est_x(signal, rho, beta, A, b, sig,
          thresh=0.0001, max_iters=10, verbosity=0):

    N = len(signal)
    admmobj = AdmmObject(N, signal, rho,
                         gaussian_x, l1_w, vector_z,
                         vector_lam, vector_alph)
    admmobj.K = 1
    admmobj.beta = beta
    admmobj.x = np.zeros((N,))
    admmobj.w = np.zeros((N,))
    admmobj.z = np.zeros((N,))
    admmobj.lam = np.random.randn(N)
    admmobj.alph = np.random.randn(N)
    admmobj.A = A
    admmobj.b = b
    admmobj.sig = sig
    G = np.eye(admmobj.N)
    for i in range(1, admmobj.N):
        G[i, i - 1] = -1
    admmobj.igginv = np.linalg.inv(np.eye(admmobj.N) + np.dot(G.transpose(), G))
    admmobj.G = G
    admmobj.Cinv = np.linalg.inv(np.dot(admmobj.A.T, admmobj.A) + (admmobj.sig * admmobj.rho) / 2 * np.eye(admmobj.L))
    admmslv = AdmmSolver(admmobj, thresh, max_iters, verbosity)
    (x, w, z, lam, alpha) = admmslv.solve()
    if verbosity > 0:
        admmslv.print_results()
    if verbosity > 1:
        admmslv.plot_residuals()
    return x
