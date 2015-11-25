function [ g ] = LRGradient(y, tX, beta)
% LRGradient(y, tX, beta)
%   Compute the gradient for the logistic regression.
%   Formally, it  is the gradient of the negative of the log-likelihood.
%
    g = tX' * (sigmoid(tX * beta) - y);
end

