function [ g ] = logisticRegGradient(y, tX, beta)
% logisticRegGradient(y, tX, beta)
%   Compute the gradient for the logistic regression.
%
    sigma = @(x) exp(x) ./ (1 + exp(x));
    g = - tX' * (y - sigma(tX * beta));
end

