function [beta] = leastSquaresGDLS(y, tX)
% leastSquaresGDLS(y, tX)
%   Least squares using gradient descent and line search to find alpha.
% 
% Solve the Least-Squares Estimate using gradient descent.
% This method assumes the data is normalized.
%

    beta0 = zeros(size(tX, 2), 1); % assuming normalized data this should work
    
    maxIters = 100;
    precision = 1e-10;
    
    beta = GDLS(y, tX, beta0, maxIters, ...
        precision, @computeMseGradient);

end

