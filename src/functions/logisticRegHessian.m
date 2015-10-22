function [ H ] = logisticRegHessian(tX, beta)
% logisticRegHessian(tX, alpha)
%   Compute the Hessian for the logistic regression.
%   Formally, it is the Hessian of the log-likelihood.
%
    sigma = @(x) exp(x) ./ (1 + exp(x));
    s = sigma(tX * beta);
    S = diag(s .* (1 - s));
    H = tX' * S * tX;
end

