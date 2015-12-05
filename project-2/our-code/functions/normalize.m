function X = normalize(X)
    X = (X - ones(size(X, 1), 1) * mean(X)) * diag(1 ./ std(X));
end
