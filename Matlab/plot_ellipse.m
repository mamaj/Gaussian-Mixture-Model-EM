function e = plot_ellipse(p, mu, sigma, c)

[V,D] = eig(sigma-1);

t = -pi:0.01:pi; t = t';
x = [D(1,1) * cos(t), D(2,2) * sin(t)] * p;
x = (V * x')';

e = plot(mu(1) + x(:,1), mu(2) + x(:,2), 'Tag', 'ell', 'Color', c, 'LineWidth', 2);

end