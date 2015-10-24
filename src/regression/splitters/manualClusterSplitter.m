function [K, clusters] = manualClusterSplitter(data)
%MANUALCLUSTERSPLITTER Return the input data in three cluster according to
%a manual heuristic of features 25 & 62
%
% The input argument `data` should at least have the following fields:
%  - train.X
%  - train.y
%  - test.X
% And so will do each returned cluster
%

    K = 3;

    % Apply manual splitting of the identified three input sources
    X25_train = data.train.X(:, 25);
    X62_train = data.train.X(:, 62);
    X25_test  = data.test.X(:, 25);
    X62_test  = data.test.X(:, 62);

    lim62 = 15.75;
    lim25 = 15.25;
    
    idx62_train = X62_train >= lim62;
    idx62_test  = X62_test  >= lim62;
    
    idx25_train = X25_train < lim25;
    idx25_test  = X25_test  < lim25;
    
    % Indexes range over 1, 2 and 3
    idx_train = idx62_train + (idx25_train & idx62_train) + 1;
    idx_test  = idx62_test  + (idx25_test  & idx62_test)  + 1;

    % Split training data & group them into clusters
    for k = 1:K
        clusters{k}.train.X = data.train.X(idx_train == k, :);
        clusters{k}.test.X  = data.test.X(idx_test == k, :);
        
        clusters{k}.train.y = data.train.y(idx_train == k, :);
    end

end

