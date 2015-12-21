function g = PLRGradient(y, tX, beta, lambda)
%
%   Compute the gradient for the penalized logistic regression.
%   Formally, it is the gradient of the negative log-likelihood + the
%   gradient of a penality term.

    % beta0 is not penalized
    pen = beta;
    pen(1) = 0;
    
    % compute the gradient
    g = tX' * (sigmoid(tX * beta) - double(y)) + 2 * lambda * pen;
end

