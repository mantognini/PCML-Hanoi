function [ H ] = LRHessian(tX, beta)
% LRHessian(tX, alpha)
%   Compute the Hessian for the logistic regression.
%   Formally, it is the Hessian of the log-likelihood.
%
    s = sigmoid(tX * beta);
    S = diag(s .* (1 - s));
    H = tX' * S * tX;
end

