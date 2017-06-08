#####
#
# General functions that are used throughout this repository
#
#####

import numpy as np

# block tridiagonal solver (see page 6 of Aravkin Paper)
def bt_solver(r,N,K):
    # is a_k (all the same)
    a = -np.eye(K)
    # initialize d and s
    d = [0]*N
    s = np.empty_like(r)
    # set first entries
    d[0] = 3 #c_n = 3*I
    s[0:K] = r[0:K]
    # forward loop
    for n in range(1,N):
        d[n] = 3 - 1/d[n-1]
        s[n*K:(n+1)*K] = r[n*K:(n+1)*K] + (1/d[n-1])*s[(n-1)*K:n*K]
        if(n == N-1):
            d[n] = 2 - 1/d[n-1] #C_N = 2*I
    # initialize solution vector
    x = np.empty_like(r)
    x[(N-1)*K:N*K] = (1/d[N-1])*s[(N-1)*K:N*K]
    # backward loop
    for n in reversed(range(0,N-1)):
        x[n*K:(n+1)*K] = (1/d[n])*(s[n*K:(n+1)*K] + x[(n+1)*K:(n+2)*K])
    return x

# Generate inverse fourier matrix
def fourier_mat(L,K,offset=0):
    F = np.zeros([L,K])
    k = np.array(range(K//2))
    l = np.array(range(0,L))
    for jj in range(0,np.size(k)):
        for ii in range(0,np.size(l)):
            F[ii,jj]      = np.cos(2*np.pi*(offset+l[ii]+1)*(k[jj])/K)
            F[ii,jj+K//2] = np.sin(2*np.pi*(offset+l[ii]+1)*(k[jj])/K)
            #F[ii,jj+K//2] = np.sin(2*np.pi*(offset+l[ii]+1)*(k[jj]+K/2)/K)
    return F

# Produces a time-dependent estimate of the noise variance
def est_noise(signal,K,N,noise_range):
    from matplotlib import mlab
    (low,high) = noise_range
    T = signal.size
    # num overlap: total win length - difference in start times
    noverlap = K - T//N
    coeffs,freqs,t = mlab.specgram(signal,NFFT=K,Fs=1,noverlap=noverlap)
    noise_est = np.mean(coeffs,axis=0)
    # Normalize between 0 and 1
    mean_min = np.min(noise_est)
    mean_max = np.max(noise_est)
    norm_est = (noise_est - mean_min) / (mean_max - mean_min)
    # Normalize shift and scale to be between 1 and 2
    scaled_est = (norm_est * (high - low)) + low
    return scaled_est

# Takes in a matrix and returns matrix of column differences
def column_diffs(A,initial_condition=True):
    if A.ndim == 1:
        A = A.reshape((1,A.size))
    (K,N) = A.shape
    diffs = np.zeros_like(A)
    diffs[:,0] = A[:,0]
    for n in range(1,N):
        diffs[:,n] = A[:,n] - A[:,n-1]
    # If we want the first column of the matrices to be the same
    if(initial_condition):
        return diffs
    # If we ONLY want differences, then we lose a column (the first column)
    else:
        return diffs[:,1:N]