function [ g ] = penLogRegGradient(y, tX, beta, lambda)
% penLogRegGradient(y, tX, beta, lambda)
%   Compute the gradient for the penalized logistic regression.
%   Formally, it is the gradient of the negative log-likelihood + the
%   gradient of a penality term.
%
    sigma = @(x) exp(x) ./ (1 + exp(x));
    
    % beta0 is not penalized
    pen = beta;
    pen(1) = 0;
    
    % compute the gradient
    g = tX' * (sigma(tX * beta) - y) + 2 * lambda * pen;
end

