function [ H ] = PLRHessian(tX, beta, lambda)
% PLRHessian(tX, beta, lambda)
%   Compute the hessian of the penalized logistic regression.
%
    % beta0 is not penalized
    penMatrix = diag(beta);
    penMatrix(1, 1) = 0;
    
    % compute the hessian
    s = sigmoid(tX * beta);
    S = diag(s .* (1 - s));
    H = tX' * S * tX + 2 * lambda * penMatrix;
end

