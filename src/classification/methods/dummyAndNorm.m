function [cluster] = dummyAndNorm(cluster)
%
    NTr = size(cluster.train.X, 2);
    discFeatures = [ 16, 25, 26, 32, 38 ];
    contFeatures = setdiff(1:NTr, discFeatures);
    
    % Normalize
    contTr = normalize(cluster.train.X(:, contFeatures));
    contTe = normalize(cluster.test.X(:, contFeatures));
    
    discTr = dummyEncoding(cluster.train.X(:, discFeatures), 5);
    discTe = dummyEncoding(cluster.test.X(:, discFeatures), 5);
    
    cluster.train.X = [contTr discTr];
    cluster.test.X = [contTe discTe];
end

