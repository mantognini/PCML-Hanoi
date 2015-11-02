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

function [beta] = gradientDescent(y, tX, beta0, alpha, maxIters, precision, gradFn)
% GRADIENTDESCENT(y, tX, beta0, alpha, maxIters, precision, gradFn)
%   Use the iterative gradient descent method to find best beta
% 
% Stop iterating either when the gradient is flat (with respect to `precision`)
% or the maximum number of iterations `maxIters` is reached.
%
% The gradient is computed by invoking gradFn(y, tX, beta).
%
% The starting point is beta0 and alpha is the step-size.
%

    D = length(beta0) - 1;

    beta = beta0;

    for k = 1:maxIters
        g = gradFn(y, tX, beta);
        beta = beta - alpha .* g;
        
        %fprintf('gradient descent k = %d; beta = ', k);
        %beta
        
        normG = g' * g / (D + 1);
        if (normG <= precision)
            %fprintf('slope is very small, stopping\n');
            break
        end
    end

end

function [ g ] = LRGradient(y, tX, beta)
% LRGradient(y, tX, beta)
%   Compute the gradient for the logistic regression.
%   Formally, it  is the gradient of the negative of the log-likelihood.
%
    g = tX' * (sigmoid(tX * beta) - y);
end

function sig = sigmoid(x)
    % Trick against numerical issue
    N = length(x);
    posIdx = find(x > 0);
    negIdx = setdiff(1:N, posIdx);
    
    xPos = x(posIdx);
    xNeg = x(negIdx);
    
    sig = zeros(N, 1);
    sig(posIdx) = ones(length(xPos), 1)  ./ (1 + exp(-xPos));
    sig(negIdx) = exp(xNeg) ./ (1 + exp(xNeg));
    
    % x(find(isnan(sig)))
end


