function [cluster] = removeDiscreteFeaturesTransformation(cluster)
%REMOVEDISCRETEFEATURESTRANSFORMATION Remove discrete features from the data 
%

    % Index of discrete features
    idx = [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61];
    
    % Remove discrete features
    cluster.train.X(:, idx) = [];
    cluster.test.X(:, idx)  = [];

end

