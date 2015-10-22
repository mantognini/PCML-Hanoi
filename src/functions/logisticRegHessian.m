function [ H ] = logisticRegHessian(tX, beta)
% logisticRegHessian(y, tX, alpha)
%   Compute the Hessian for the logistic regression.
%
    sigma = @(x) exp(x) ./ (1 + exp(x));
    s = sigma(tX * beta);
    S = diag(s .* (1 - s));
    H = tX' * S * tX;
end

