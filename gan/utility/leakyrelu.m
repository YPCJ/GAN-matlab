function y = leakyrelu(x)
   
% slow version
%    y = x;
%    y(y<=0) = 0;

% faster version: make use of matrix opts
   mask = single(x>0) + 0.1*single(x<0);
   y = x.*mask;
end
