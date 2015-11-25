function [beta] = leastSquaresGD(y, tX, alpha)
% LEASTSQUARESGD(y, tX, alpha)
%   Least squares using gradient descent
% 
% Solve the Least-Squares Estimate using gradient descent.
% This method assumes the data is normalized.
%
% The step-size is alpha.
%

    beta0 = zeros(size(tX, 2), 1); % assuming normalized data this should work
    
    maxIters = 1000;
    precision = 1e-10;
    
    beta = gradientDescent(y, tX, beta0, alpha, maxIters, precision, @computeMseGradient);

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

function [g] = computeMseGradient(y, tX, beta)
% COMPUTEMSEGRADIENT(y, tX, beta)
%   Compute the gradient of Mean Square Error cost function
%

    N = length(y);
    e = y - tX * beta;
    g = - 1 / N * tX' * e;

end



