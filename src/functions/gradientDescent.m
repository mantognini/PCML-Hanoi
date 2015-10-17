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
