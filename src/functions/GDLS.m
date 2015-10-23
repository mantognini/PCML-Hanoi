function [beta] = GDLS(y, tX, beta0, maxIters, precision, gradFn)
% GDLS(y, tX, beta0, maxIters, precision, gradFn)
%   Use the iterative gradient descent method to find best beta using line
%   search to compute alpha at eat step.
% 
% Stop iterating either when the gradient is flat (with respect to `precision`)
% or the maximum number of iterations `maxIters` is reached.
%
% The gradient is computed by invoking gradFn(y, tX, beta).
%
% The starting point is beta0.
%

    D = length(beta0) - 1;
    beta = beta0;
    g = gradFn(y, tX, beta);

    for k = 1:maxIters
        % Line search
        alpha = 1;
        sigma = 10^(-4);
        for alphaSearchStep = 1:10
            % Try a step
            betaNew = beta - alpha .* g;
            gNew = gradFn(y, tX, betaNew);

            if norm(gNew) < (1 - sigma * alpha) * norm(g)
                beta = betaNew;
                g = gNew;
                break;
            end

            % otherwise, reduce alpha
            alpha = alpha / 2;
        end
    
        % Test stop condition
        normG = g' * g / (D + 1);
        if (normG <= precision)
            break
        end
    end
end
