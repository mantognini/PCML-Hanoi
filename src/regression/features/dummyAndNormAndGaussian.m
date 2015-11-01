function [cluster] = dummyAndNormAndGaussian(cluster)
%
    DTr = size(cluster.train.X, 2);
    NTr = size(cluster.train.X, 1);
    discFeatures = [ 16, 25, 26, 32, 38 ];
    contFeatures = setdiff(1:DTr, discFeatures);
    
    % Apply poisson -> Gaussian transformation
    cluster.train.X = abs(cluster.train.X).^(1/2);
    cluster.test.X = abs(cluster.test.X).^(1/2);
    
    % Normalize
    contTr = normalize(cluster.train.X(:, contFeatures));
    contTe = normalize(cluster.test.X(:, contFeatures));
    
    discTr = dummyEncoding(cluster.train.X(:, discFeatures), 5);
    discTe = dummyEncoding(cluster.test.X(:, discFeatures), 5);
    
    cluster.train.X = [contTr discTr];
    cluster.test.X = [contTe discTe];
end

