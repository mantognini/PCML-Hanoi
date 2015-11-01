function [ H ] = PLRHessian(tX, beta, lambda)
% PLRHessian(tX, beta, lambda)
%   Compute the hessian of the penalized logistic regression.
%
    sigma = @(x) exp(x) ./ (1 + exp(x));
    
    % beta0 is not penalized
    penMatrix = diag(beta);
    penMatrix(1, 1) = 0;
    
    % compute the hessian
    s = sigma(tX * beta);
    S = diag(s .* (1 - s));
    H = tX' * S * tX + 2 * lambda * penMatrix;
end

