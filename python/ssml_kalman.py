import numpy as np

def kalman_smooth(obs,eps,params):

    (M,Q,R) = obs

    x = np.zeros(len(M))
    x_k_ = np.zeros(len(M))
    x_k = np.zeros(len(M))
    sig_k_ = np.zeros(len(M))
    sig_k = np.zeros(len(M))


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

    for i in range(len(x)):
        # update one step estimates
        x_k_[i] = gamma + phi* x_k[i-1]
        sig_k_[i] = (phi**2)*sig_k[i-1] + sigv

        # update gain
        C_k = sig_k_[i]/((h**2)*sig_k_[i] + sigg)

        # update posterior mode
        q = Q[i]
        m = M[i]
        r = R[i,:]

        diff = eps + 1
        x_k[i] = x_k_[i]
        while(diff > eps):

            p_k = np.exp(mu + eta * x_k[i])/ \
                  (1+np.exp(mu + eta * x_k[i]))

            lam_k = np.zeros_like(r)
            lam_k[0] = np.exp(psi + g*x_k[i])
            for S in range(1,len(c)):
                lam_k[S] = np.exp(psi + g*x_k[i] + \
                    np.dot(c[:S],r[:S]))
            for j in range(len(c),len(lam_k)):
                lam_k[j] = np.exp(psi + g*x_k[i] + \
                    np.dot(c,r[j-len(c):j]))


            rho = -x_k[i] + x_k_[i] + \
                C_k*(h*(q - delta - h*x_k_[i]) + \
                eta * sigg * (m - p_k)) + \
                C_k * sigg * g * np.sum(r-lam_k*Del)

            d_rho = -1 - C_k*sigg*((eta**2)*p_k*(1-p_k) + \
                (g**2)*np.sum(lam_k)*Del)

            x_k[i] = x_k[i] - rho/d_rho
            diff = np.abs(rho/d_rho)

        x[i] = x_k[i]

        p_k = np.exp(mu + eta * x_k[i])/ \
              (1+np.exp(mu + eta * x_k[i]))

        lam_k = np.zeros_like(r)
        lam_k[0] = np.exp(psi + g*x_k[i])
        for S in range(1,len(c)):
            lam_k[S] = np.exp(psi + g*x_k[i] + \
                np.dot(c[:S],r[:S]))
        for j in range(len(c),len(lam_k)):
            lam_k[j] = np.exp(psi + g*x_k[i] + \
                np.dot(c,r[j-len(c):j]))

        sig_k[i] = 1/(1/sig_k_[i] + (h**2)/sigg + (eta**2)*p_k*(1-p_k) + (g**2)*np.sum(lam_k)*Del)

        for i in reversed(range(len(x)-1)):
            A_k = phi*(sig_k[i]/sig_k_[i+1])
            x[i] = x_k[i] + A_k*(x[i+1]-x_k_[i+1])

    return x
