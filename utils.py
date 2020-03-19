import numpy as np
import matplotlib.pyplot as plt


class PlotGmm:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.el = []
        self.setup_ax()

    def setup_ax(self):
        self.ax.set_aspect('equal')
        self.ax.grid()
        plt.ion()
        plt.show()

    def plot_ellipse(self, mu, cov, pi, **kwargs):
        t = np.linspace(0, 2*np.pi, 100)
        l, V = np.linalg.eigh(cov)
        xbar = np.zeros((2, t.size))
        xbar[0, :] = l[0] * np.cos(t) * pi 
        xbar[1, :] = l[1] * np.sin(t) * pi
        x = V @ xbar + mu.reshape([-1,1])
        return self.ax.plot(*x, **kwargs)[0]

    def plot_data(self, x, z, mu_list, cov_list, pi_list):
        self.ax.scatter(*x.T, s=4, c=np.array(['b', 'r'])[z])
        for mu, cov, pi, c in zip(mu_list, cov_list, pi_list, ['b', 'r']):
            self.plot_ellipse(mu, cov, pi, color=c)
        plt.show()
        plt.pause(1)


    def update_plot(self, mu_list, cov_list, pi_list, it, nll=None):
        # remove previously plotted ellipsis
        for e in self.el:
            e.remove()
        self.el.clear()
    
        for mu, cov, pi in zip(mu_list, cov_list, pi_list):
            self.el.append(self.plot_ellipse(mu, cov, pi, color='g', ls='--'))
        if nll is not None:
            self.ax.set_title(f'{it} - NLL: {nll:.6f}')
        else:
            self.ax.set_title(f'{it}')
        plt.pause(0.001)


class DataGenerator:
    def __init__(self, n=2000, mu=None, cov=None, pi=[1., 2.]):
        self.n = n
        
        if mu is None:
            self.mu = np.array(
                [
                    [-2., -2.],
                    [1., 1.]
                ], 
                dtype=np.float32
            )
        else:
            self.mu = np.array(mu)

        if cov is None:
            self.cov = np.array(
                [
                    np.diag([2., 5.]),
                    np.eye(2) * 1.
                ],
                dtype=np.float32
            )
        else:
            self.cov = np.array(cov)

        pi = np.array(pi)
        self.pi = pi / pi.sum()

    def generate_data(self, dtype=np.float32, seed=1):
        rng = np.random.RandomState(seed=seed)
        ncomps = rng.multinomial(self.n, self.pi)
        z = [[i] * ncomp for i, ncomp in enumerate(ncomps)]
        z = np.hstack(z)
        x = []
        for ncomp, m, s in zip(ncomps, self.mu, self.cov):
            x.append(rng.multivariate_normal(m, s, size=ncomp))
        x = np.vstack(x)
        return x.astype(dtype), z.astype(np.int)

