function [y, x_hat, istd] = batchNorm_forward(x, gamma, beta)

mu = mean(x, 2);
istd = 1./sqrt( var(x, 1, 2) + 10^-5 );

x_hat = bsxfun(@minus, x, mu);
x_hat = bsxfun(@times, x_hat, istd);

y = bsxfun(@times, x_hat, gamma);
y = bsxfun(@plus, y, beta);

istd = istd';
end