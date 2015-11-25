function beta = logisticRegression(y,tX,alpha)
% logisticRegression(y,tX,alpha)
%   Compute the logistic regression using Gradient Descent method.
%
    % initialization
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Gradient descent
    customGradient = @(y, tX, beta) LRGradient(y, tX, beta);
    beta = gradientDescent(y, tX, beta0, alpha, maxIters, precision, customGradient);
end

