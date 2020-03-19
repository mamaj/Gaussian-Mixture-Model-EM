import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mnorm
from scipy.special import logsumexp
from utils import PlotGmm, DataGenerator

## Generate Data
n=1000
dg = DataGenerator(n)
x, z = dg.generate_data()

## Plot GMM
plot_gmm = PlotGmm()
plot_gmm.plot_data(x, z, dg.mu, dg.cov, dg.pi)

## Initial Parameters
khat = 2
pi_hat = np.random.uniform(size=khat)
pi_hat /= pi_hat.sum()
mu_hat = np.random.normal(size=(khat,2))
cov_hat = np.zeros((khat, 2, 2))
for s in cov_hat:
    np.fill_diagonal(s, np.random.uniform(size=(2,)))

# Collapse one component
# pi_hat[-1] = 0.01
# pi_hat /= pi_hat.sum()
# mu_hat[-1] = x[0]
# cov_hat[-1] = np.eye(2) * 1e-20
# ax.plot(*mu_hat[-1], ms=20, marker='o', mfc="None", lw=3)

## Run EM
niter = 100
for it in range(niter):

    # E step
    log_joint = np.zeros((n, khat))
    for i, (m, s, p) in enumerate(zip(mu_hat, cov_hat, pi_hat)):
        log_joint[:, i] = mnorm.logpdf(x, m, s) +  np.log(p)
    log_marginal = logsumexp(log_joint, axis=1, keepdims=True) 
    nll =  - log_marginal.mean()
    log_post = log_joint - log_marginal
    comp_resp = logsumexp(log_post, axis=0)

    plot_gmm.update_plot(mu_hat, cov_hat, pi_hat, it, nll)

    # M step
    for i in range(khat):
        mu_hat[i, :] = np.exp(log_post[:, i] - comp_resp[i]) @ x
        cov_hat[i, :] =  (x - mu_hat[i]).T @ np.diag(np.exp(log_post[:, i] - comp_resp[i])) @ (x - mu_hat[i])
        pi_hat[i] =  np.mean(np.exp(log_post[:, i]))

print(pi_hat, pi)
plt.show(block=True)