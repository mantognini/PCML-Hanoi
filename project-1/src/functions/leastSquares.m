function [beta] = leastSquares(y, tX)
% LEASTSQUARES(y, tX)
%   Solve Least-Squares Estimate using equations
%

    % Solve equations using Matlab's QR decompsition with \
    beta = (tX' * tX) \ (tX' * y);

end

