function [beta] = leastSquares(y, tX)
% LEASTSQUARES(y, tX)
%   Solve Least-Squares Estimate using normal equations
%

    % Solve equations using Matlab's QR decomposition with \
    beta = (tX' * tX) \ (tX' * y);

end

