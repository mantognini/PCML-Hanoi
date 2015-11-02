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

function [ g ] = PLRGradient(y, tX, beta, lambda)
% PLRGradient(y, tX, beta, lambda)
%   Compute the gradient for the penalized logistic regression.
%   Formally, it is the gradient of the negative log-likelihood + the
%   gradient of a penality term.
%
    % beta0 is not penalized
    pen = beta;
    pen(1) = 0;
    
    % compute the gradient
    g = tX' * (sigmoid(tX * beta) - y) + 2 * lambda * pen;
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
end

