function [enrichedX] = dummyEncoding(X, K)
% dummyEncoding(X)
%   Each feature with less or equal than K distinct values is dummy encoded.
%
    enrichedX = [];
    for i = 1:size(X, 2)
        feature = X(:, i);
        u = unique(feature);
        uSize = length(u);
        if uSize <= K
            for j = 1:uSize
                newFeature = (feature == u(j));
                enrichedX = [enrichedX newFeature];
            end
        else
            enrichedX = [enrichedX feature];
        end
    end
end