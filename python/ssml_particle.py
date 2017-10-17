import numpy as np
from scipy.stats import chi2

def particle_smooth(obs,params,num_p=100,
    sparse=False,checks=False,forward_only=False):

    gamma = params[0]
    sigv = params[2]

    if sparse:
        p = params[13]
        sigchi = params[14]

    (M,Q,R) = obs
    N = len(M)

    particles = np.zeros((N,num_p))
    uparticles = np.zeros((N,num_p))
    W = np.zeros((N,num_p))

    ######### Forward filter
    # Initialize particles
    if checks: print('Position 0')
    samples = gen_sample(np.zeros(num_p),params,sparse)
    weights = get_likelihood(0,samples,obs,params,checks)
    W[0,:] = weights/np.sum(weights)
    uparticles[0,:] = samples
    particles[0,:] = resample(samples,W[0,:],checks)
    for n in range(1,N):
        if checks:
            print('Position %i'%n)
        samples = gen_sample(particles[n-1,:],params,sparse)
        weights = get_likelihood(n,samples,obs,params,checks)
        #print('Weights: ' + str(weights))
        W[n,:] = weights/np.sum(weights)
        uparticles[n,:] = samples
        particles[n,:] = resample(samples,W[n,:],checks)
        if checks: print(np.mean(particles[n,:]))

    if forward_only: return np.mean(particles,axis=1)
    ######## Backward smoother
    smoothed = np.zeros(N)
    smoothed[-1] = np.mean(particles[-1,:])
    for n in reversed(range(N-1)):
        x_next = smoothed[n+1]
        x = uparticles[n,:]
        if not sparse:
            f = 1/np.sqrt(2*np.pi*sigv)* \
            np.exp(-0.5*(x_next - gamma - x)**2/sigv)
        else:
            f = []
            for uparticle in x:
                if uparticle >= x_next:
                    f.append(1-p)
                #elif uparticle > x_next:
                #    f.append(0)
                else:
                    f.append(p*chi2.pdf(x_next-uparticle,2,scale=np.sqrt(sigchi)))
            f = np.asarray(f)
        weights = (W[n,:]*f)/np.sum(W[n,:]*f)
        smoothed[n] = np.dot(x,weights)


    return smoothed


def get_likelihood(i,samples,obs,params,checks):
    (M,Q,R) = obs

    q = Q[i]
    m = M[i]
    r = R[i,:]

    gamma = params[0]
    phi = params[1]
    sigv = params[2]
    delta = params[3]
    h = params[4]
    sigg = params[5]
    mu = params[6]
    eta = params[7]
    psi = params[8]
    g = params[9]
    c = params[10]
    Del = params[11]
    J = params[12]

    lq = 1/(np.sqrt(2*np.pi*sigg))* \
        np.exp(-0.5*(h*samples + delta - 1)**2/sigg)

    p = np.exp(mu + eta*samples)
    lm = (p/(1+p))**m * (1/(1+p))**(1-m)

    lr = []
    for x in samples:
        lam = np.zeros_like(r)
        lam[0] = np.exp(psi + g*x)
        for S in range(1,len(c)):
            lam[S] = np.exp(psi + g*x + \
                np.dot(c[:S],r[:S]))
        for j in range(len(c),len(lam)):
            lam[j] = np.exp(psi + g*x + \
                np.dot(c,r[j-len(c):j]))
        lr.append(np.exp(np.sum(np.log(lam)*r - lam*Del)))
    lr = np.asarray(lr) + 10**(-60)

    likelihood = lq*lm*lr
    if((likelihood == 0).all == True): print("Degenerate likelihood")

    return lq*lm*lr + 10**(-60)

def resample(samples,weights,checks):
    num_samples = len(samples)
    positions = (np.arange(num_samples) + np.random.uniform()) / num_samples
    cum_sum = np.cumsum(weights)
    resamples = []

    i, j = 0,0
    while i < num_samples:
        if positions[i] <= cum_sum[j]:
            resamples.append(samples[j])
            i += 1
        else:
            j+= 1
    return np.asarray(resamples)


def gen_sample(start,params,sparse):

    gamma = params[0]
    sigv = params[2]

    if sparse:
        p = params[13]
        sigchi = params[14]

    if not sparse:
        return start + np.random.randn(len(start))*np.sqrt(sigv) + gamma
    else:
        count = 0
        new_samples = []
        for sample in start:
            if np.random.uniform() > p:
                new_samples.append(sample)
            else:
                count += 1
                new_samples.append(sample + \
                    sigchi*(np.random.chisquare(2)))
        return np.asarray(new_samples)