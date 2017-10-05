#####
#
# Applies the ADMM code to solve the state space model of learning problem.
#
#####

import numpy as np
from framework import AdmmObject, AdmmSolver
from updates import ssml_x, ssml_x_grad, ssml_x_parallel, ssml_x_grad_parallel, vector_z, least_sq_w, l1_w
from updates import vector_lam, vector_alph


def ssml(observations, rho, params,
         thresh=0.0001, max_iters=10, verbosity=0,
         history=False, parallel=False, approx=False,
         sparse=False):
    # Number of trials
    N = len(observations[0])
    if sparse:
        admmobj = AdmmObject(N, observations, rho,
                             ssml_x_parallel, vector_z,
                             l1_w, vector_lam,
                             vector_alph)
        admmobj.beta = params[13]
    else:
        admmobj = AdmmObject(N, observations, rho,
                             ssml_x_parallel, vector_z,
                             least_sq_w, vector_lam,
                             vector_alph)
        admmobj.beta = 1
    # Augmented Lagrange parameter
    admmobj.rho = rho
    # A bunch of Parameters for ssml problem
    admmobj.gamma = params[0]
    admmobj.phi = params[1]
    admmobj.sigv = params[2]
    admmobj.delta = params[3]
    admmobj.h = params[4]
    admmobj.sigg = params[5]
    admmobj.mu = params[6]
    admmobj.eta = params[7]
    admmobj.psi = params[8]
    admmobj.g = params[9]
    admmobj.c = params[10]
    admmobj.Del = params[11]
    admmobj.J = params[12]
    # x - Estimate of learning state
    admmobj.x = np.zeros((admmobj.N,))
    # z - enforces relationship between x and w
    admmobj.z = np.zeros((admmobj.N,))
    # w - penalty term
    admmobj.w = np.zeros((admmobj.N,))
    admmobj.lam = np.ones((admmobj.N,))
    admmobj.alph = np.ones((admmobj.N,))
    # Extra variables used in z update
    if sparse:
        G = np.eye(admmobj.N)
        for i in range(1, admmobj.N):
            G[i, i - 1] = -1
        admmobj.igginv = np.linalg.inv(np.eye(admmobj.N) + np.dot(G.transpose(), G))
        admmobj.G = G
    else:
        G = np.eye(admmobj.N)
        for i in range(1, admmobj.N):
            G[i, i - 1] = -admmobj.phi
        admmobj.igginv = np.linalg.inv(np.eye(admmobj.N) + np.dot(G.transpose(), G))
        admmobj.G = G

    admmslv = AdmmSolver(admmobj, thresh, max_iters, verbosity, history)
    (x, w, z, lam, alph) = admmslv.solve()
    if verbosity > 0:
        admmslv.print_results()
    if verbosity > 1:
        admmslv.plot_residuals()
    if history:
        return x, w, z, lam, alph
    else:
        return x
