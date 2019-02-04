function [dx, dgamma, dbeta] = batchNorm_backpass(x_hat, istd, dout, gamma)

N=size(x_hat,2);
gamma = gamma';

dbeta = sum(dout);
dgamma = sum(x_hat'.*dout);

dx = bsxfun(@times,(gamma.*istd/N), (N*dout - bsxfun(@plus,bsxfun(@times,x_hat', dgamma),dbeta)));

dbeta = dbeta';
dgamma = dgamma';
end