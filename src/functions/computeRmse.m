function error = computeRmse(e)
% computeRmse(e)
%   Compute the rmse error of e.
%   Note that e is expected to be a column vector.
%
    N = size(e, 1);
    error = sqrt(e'*e / N);
end