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

