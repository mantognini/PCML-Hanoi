function [cluster] = keepFeature25And62Transformation(cluster)
%REMOVEDISCRETEFEATURESTRANSFORMATION Remove discrete features from the data 
%

    D = size(cluster.train.X, 2);

    idx = setdiff(1:D, [25 62]);
    
    % Remove all other features
    cluster.train.X(:, idx) = [];
    cluster.test.X(:, idx)  = [];

end

