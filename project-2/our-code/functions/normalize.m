function X = normalize(X)
%
% Normalize X with 0 mean, variance of 1
    X = (X - ones(size(X, 1), 1) * mean(X)) * diag(1 ./ std(X));
end
