function beta = LRLS(y,tX)
% Logistic regression using Gradient Descent method + Line search
%
    % initialization
    [~, M] = size(tX);
    beta0 = zeros(M, 1); % aka beta0
    precision = 1^(-10);
    maxIters = 100;
    
    % Gradient descent
    customGradient = @(y, tX, beta) LRGradient(y, tX, beta);
    beta = GDLS(y, tX, beta0, maxIters, precision, customGradient);
end

