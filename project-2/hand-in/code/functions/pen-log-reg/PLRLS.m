function beta = PLRLS(y, tX, lambda)
%
% Penalized logistic regression with line search

    % initialization
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Gradient descent with Line Search
    customGradient = @(y, tX, beta) PLRGradient(y, tX, beta, lambda);
    beta = GDLS(y, tX, beta0, maxIters, precision, customGradient);
end

