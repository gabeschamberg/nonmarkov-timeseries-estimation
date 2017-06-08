#####
#
# Formulate a latent time series estimation problem as an optimization problem
# to be solved with ADMM
#
#####
#
# AdmmObject - object to track all the details of the problem, such as the
#       update functions, the parameters, the observations, and the current
#       estimates of X, Z, and W. Essentially contains all the details of
#       the problem formulation
#
# AdmmSolver - takes in an AdmmObject and obtains the estimates. AdmmSolvers
#       keep track of the details of actually executing the ADMM algorithm, such
#       as the threshold for the residuals and the maximum number of iterations.
#       Also handles tracking and reporting of the residuals, runtimes, etc.
#
#####

import numpy as np
import time
import sys
from helpers import column_diffs


# Object to keep track of parameters, current ADMM estimates,
# and specify specific measurement model and pen function
class AdmmObject:
    def __init__(self, N, obs, rho,
                 update_x, update_w, update_z,
                 update_lam, update_alph):
        self.N = N
        self.obs = obs
        self.rho = rho
        self.update_x = update_x
        self.update_w = update_w
        self.update_z = update_z
        self.update_lam = update_lam
        self.update_alph = update_alph


class AdmmSolver:
    def __init__(self, obj, thresh, max_iters, verbosity, history=False):
        # obj - the ADMM object being solved
        self.start_time = time.time()
        self.runtime = 0
        self.obj = obj
        self.eps = thresh
        self.max_iters = max_iters
        self.v = verbosity
        self.history = history
        # keep track of average time to update each variable
        self.avgx = 0
        self.avgz = 0
        self.avgw = 0
        self.avgl = 0
        self.avga = 0
        # keep track of dual residuals
        self.s1 = []
        self.s2 = []
        # keep track of primal residuals
        self.r1 = []
        self.r2 = []
        # keep track of thresholds
        self.eps_s1 = []
        self.eps_s2 = []
        self.eps_r1 = []
        self.eps_r2 = []
        # create difference matrix
        self.A = np.eye(self.obj.N)
        for n in range(1, self.obj.N):
            self.A[n - 1, n] = -1

    def solve(self):
        converged = False
        iters = 0
        if self.v > 0:
            print("Beginning ADMM Iterations")
        xs = [self.obj.x]
        ws = [self.obj.w]
        zs = [self.obj.z]
        lams = [self.obj.lam]
        alphs = [self.obj.lam]
        # Continue updating variables until converged
        while not converged and iters < self.max_iters:
            # x update
            start = time.time()
            x_ = np.copy(self.obj.x)
            self.obj.update_x(self.obj)
            xtime = time.time() - start
            xs.append(self.obj.x)
            if (self.v > 3):
                print("x updated in " + str(xtime) + " seconds")
                sys.stdout.flush()
            # w update
            start = time.time()
            w_ = np.copy(self.obj.w)
            self.obj.update_w(self.obj)
            wtime = time.time() - start
            ws.append(self.obj.w)
            if (self.v > 3):
                print("w updated in " + str(wtime) + " seconds")
                sys.stdout.flush()
            # z update
            start = time.time()
            z_ = np.copy(self.obj.z)
            self.obj.update_z(self.obj)
            ztime = time.time() - start
            zs.append(self.obj.z)
            if (self.v > 3):
                print("z updated in " + str(ztime) + " seconds")
                sys.stdout.flush()
            # lambda update
            start = time.time()
            lam_ = np.copy(self.obj.lam)
            self.obj.update_lam(self.obj)
            ltime = time.time() - start
            lams.append(self.obj.lam)
            if (self.v > 3):
                print("lambda updated in " + str(ltime) + " seconds")
                sys.stdout.flush()
            # alpha update
            start = time.time()
            alph_ = np.copy(self.obj.alph)
            self.obj.update_alph(self.obj)
            atime = time.time() - start
            alphs.append(self.obj.alph)
            if (self.v > 3):
                print("alpha updated in " + str(atime) + " seconds")
                sys.stdout.flush()
            # Update residuals and check for convergence
            converged = self.update_residuals(x_, z_, w_, lam_, alph_)
            # Update iteration count
            iters = iters + 1
            # Update average runtimes
            self.avgx = self.avgx * ((iters - 1) / iters) + xtime * (1 / iters)
            self.avgw = self.avgw * ((iters - 1) / iters) + wtime * (1 / iters)
            self.avgz = self.avgz * ((iters - 1) / iters) + ztime * (1 / iters)
            self.avgl = self.avgl * ((iters - 1) / iters) + ltime * (1 / iters)
            self.avga = self.avga * ((iters - 1) / iters) + atime * (1 / iters)
        self.runtime = time.time() - self.start_time
        self.iters = iters
        if self.history:
            return (xs, ws, zs, lams, alphs)
        else:
            return (self.obj.x, self.obj.w, self.obj.z, self.obj.lam, self.obj.alph)

    def update_residuals(self, x_, z_, w_, lam_, alph_):
        # Update the residual norms
        s1norm = np.linalg.norm(self.obj.rho * (self.obj.w - w_).dot(self.A.T))
        self.s1.append(s1norm)
        s2norm = np.linalg.norm(self.obj.rho * (self.obj.z - z_))
        self.s2.append(s2norm)
        r1norm = np.linalg.norm(self.obj.x - self.obj.z)
        self.r1.append(r1norm)
        r2norm = np.linalg.norm(self.obj.w - self.obj.z.dot(self.A))
        self.r2.append(r2norm)
        # Update the convergence thresholds
        eps_abs = self.eps * np.sqrt(self.obj.x.size)
        eps_s1 = eps_abs + \
                 self.eps * max([np.linalg.norm(x_), np.linalg.norm(z_)])
        self.eps_s1.append(eps_s1)
        eps_s2 = eps_abs + \
                 self.eps * max([np.linalg.norm(w_), np.linalg.norm(z_.dot(self.A))])
        self.eps_s2.append(eps_s2)
        eps_r1 = eps_abs + self.eps * np.linalg.norm(lam_)
        self.eps_r1.append(eps_r1)
        eps_r2 = eps_abs + self.eps * np.linalg.norm(alph_.dot(self.A.T))
        self.eps_r2.append(eps_r2)
        if (s1norm < eps_s1 and s2norm < eps_s2 and
                    r1norm < eps_r1 and r2norm < eps_r2):
            return True
        else:
            return False

    def plot_residuals(self):
        import matplotlib.pyplot as plt
        t = range(1, self.iters + 1)
        plt.figure(figsize=(15, 5))
        plt.semilogy(t, self.s1, 'r', label='$s_1^i$')
        plt.semilogy(t, self.s2, 'b', label='$s_2^i$')
        plt.semilogy(t, self.r1, 'g', label='$r_1^i$')
        plt.semilogy(t, self.r2, 'm', label='$r_2^i$')
        plt.semilogy(t, self.eps_s1, 'r--', label='$\epsilon_1^{dual}$')
        plt.semilogy(t, self.eps_s2, 'b--', label='$\epsilon_2^{dual}$')
        plt.semilogy(t, self.eps_r1, 'g--', label='$\epsilon_1^{pri}$')
        plt.semilogy(t, self.eps_r2, 'm--', label='$\epsilon_2^{pri}$')
        plt.legend(ncol=2)
        plt.xlabel('Iteration ($i$)')
        plt.ylabel('Residual Norm (log scale)')
        plt.title('Residuals and Thresholds')
        plt.xlim([1, self.iters])

    def print_results(self):
        print("\n")
        print("Total runtime: " + str(self.runtime) + " seconds")
        print("Total number of iterations: " + str(self.iters))
        print("Norm of differences between x and z: " + \
              str(np.sum(np.linalg.norm(self.obj.x - self.obj.z))))
        Z = column_diffs(self.obj.z)
        print("Norm of differences between w and z differences: " + \
              str(np.sum(np.linalg.norm(self.obj.w - Z))))
        print("Average Update Times:")
        print("x - " + str(self.avgx) + " seconds")
        print("w - " + str(self.avgw) + " seconds")
        print("z - " + str(self.avgz) + " seconds")
        print("lambda - " + str(self.avgl) + " seconds")
        print("alpha - " + str(self.avga) + " seconds")
        print("---------------------------------------\n")
