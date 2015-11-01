function beta = PLRLSNewton(y,tX,lambda)
% penalized logistic regression with line search and newton
%
    % initialization
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Newton's method is equivalent to gradient descent with a custom
    % gradient taking into account second-order information
    customGradient = @(y, tX, beta) PLRHessian(tX, beta, lambda) \ ...
        PLRGradient(y, tX, beta, lambda);
    beta = GDLS(y, tX, beta0, maxIters, precision, customGradient);
end

