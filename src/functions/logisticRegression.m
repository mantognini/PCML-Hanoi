function beta = logisticRegression(y,tX,alpha)
% LOGISTICREGRESSION(y, tX, alpha)
%   Compute the logistic regression using Newton's method.
%
    % Newton method
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Newton's method is equivalent to gradient descent with a custom
    % gradient taking into account second-order information
    customGradient = @(y, tX, beta) logisticRegHessian(tX, beta) \ logisticRegGradient(y, tX, beta);
    beta = gradientDescent(y, tX, beta0, alpha, maxIters, precision, customGradient);
end

