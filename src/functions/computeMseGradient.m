function [g] = computeMseGradient(y, tX, beta)
% COMPUTEMSEGRADIENT(y, tX, beta)
%   Compute the gradient of Mean Square Error cost function
%

    N = length(y);
    e = y - tX * beta;
    g = - 1 / N * tX' * e;

end

