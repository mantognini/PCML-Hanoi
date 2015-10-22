function [ g ] = logisticRegGradient(y, tX, beta)
% logisticRegGradient(y, tX, beta)
%   Compute the gradient for the logistic regression.
%   Formally, it  is the gradient of the negative of the log-likelihood.
%
    sigma = @(x) exp(x) ./ (1 + exp(x));
    g = tX' * (sigma(tX * beta) - y);
end

