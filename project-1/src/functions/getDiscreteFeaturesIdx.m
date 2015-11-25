function [ discreteIdx ] = getDiscreteFeaturesIdx(X, K)
    nbFeatures = size(X, 2);
    discreteIdx = false(nbFeatures, 1);
    for i = 1:nbFeatures
        discreteIdx(i) = (length(unique(X(:, i))) <= K);
    end
end