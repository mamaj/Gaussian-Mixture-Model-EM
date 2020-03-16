import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mnorm


def plot_ellipse(mu, cov, pi, ax, **kwargs):
    t = np.linspace(0, 2*np.pi, 100)
    l, V = np.linalg.eigh(cov)
    xbar = np.zeros((2, t.size))
    xbar[0, :] = l[0] * np.cos(t) * pi 
    xbar[1, :] = l[1] * np.sin(t) * pi
    x = V @ xbar + mu.reshape([-1,1])
    return ax.plot(*x, **kwargs)[0]


def update_plot():
    for e in el:
        e.remove()
    el.clear()
    for m, s, p in zip(mu_hat, cov_hat, pi_hat):
        el.append(plot_ellipse(m, s, p, ax, color='g', ls='--'))
    ax.set_title(f'{it} - NLL: {nll:.6f}')
    plt.pause(0.001)
    print(f'{it + 1}: {nll:.8f}')


## Parameters
n = 1000
mu = np.array([
    [-2, -2], 
    [1, 1]
    ])
cov = np.array([
    np.eye(2) * 3,
    np.eye(2) * 1
    ])
pi = np.array([1, 2])
pi = pi / pi.sum()
k = pi.shape[0]

## Generate Data
ncomps = np.random.multinomial(n, pi)
z = [[i] * ncomp for i, ncomp in enumerate(ncomps)]
z = np.hstack(z)
x = []
for ncomp, m, s in zip(ncomps, mu, cov):
    x.append(np.random.multivariate_normal(m, s, size=ncomp))
x = np.vstack(x)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(*x.T, s=4, c=np.array(['b', 'r'])[z])
for m, s, p, c in zip(mu, cov, pi, ['b', 'r']):
    plot_ellipse(m, s, p, ax, color=c)
plt.ion()
plt.show()


## Initial Parameters
khat = 2
pi_hat = np.random.uniform(size=khat)
pi_hat /= pi_hat.sum()
mu_hat = np.random.normal(size=(khat,2))
cov_hat = np.zeros((khat, 2, 2))
for s in cov_hat:
    np.fill_diagonal(s, np.random.uniform(size=(2,)))
el = []

## Collapse one component
# pi_hat[-1] = 0.01
# pi_hat /= pi_hat.sum()
# mu_hat[-1] = x[0]
# cov_hat[-1] = np.eye(2) * 1e-20
# ax.plot(*mu_hat[-1], ms=20, marker='o', mfc="None", lw=3)

## Run EM
niter = 100
for it in range(niter):

    # E step
    resp = np.zeros((n, khat))
    for i, (m, s, p) in enumerate(zip(mu_hat, cov_hat, pi_hat)):
        resp[:, i] = mnorm.pdf(x, m, s) * p
    nll = - np.log(resp.sum(1)).mean()
    resp = resp / resp.sum(1, keepdims=True)

    update_plot()

    # M step
    for i in range(khat):
        sum_resp = np.sum(resp[:, i])
        mu_hat[i, :] = resp[:, i] @ x / sum_resp
        cov_hat[i, :] =  (x - mu_hat[i]).T @ np.diag(resp[:, i]) @ (x - mu_hat[i]) / sum_resp
        pi_hat[i] =  np.mean(resp[:, i])

print(pi_hat, pi)
plt.show(block=True)