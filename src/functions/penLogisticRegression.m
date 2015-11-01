function beta = penLogisticRegression(y,tX,alpha,lambda)
% penLogisticRegression(y,tX,alpha,lambda)
%   Compute the penalized logistic regression using Gradient descent.
%
    % initialization
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Gradient Descent
    customGradient = @(y, tX, beta) PLRGradient(y, tX, beta, lambda);
    beta = gradientDescent(y, tX, beta0, alpha, maxIters, precision, customGradient);
end

