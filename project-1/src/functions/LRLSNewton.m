function beta = LRLSNewton(y,tX)
%   Logistic regression using Newton method + Line Search.
%
    % initialization
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Gradient descent
    customGradient = @(y, tX, beta) LRHessian(tX, beta) \ ...
        LRGradient(y, tX, beta);
    beta = GDLS(y, tX, beta0, maxIters, precision, customGradient);
end

