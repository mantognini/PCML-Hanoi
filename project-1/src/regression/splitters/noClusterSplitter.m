function [K, clusters] = noClusterSplitter(data)
%NOCLUSTERSPLITTER Return the input data in one cluster only
%
% The input argument `data` should at least have the following fields:
%  - train.X
%  - train.y
%  - test.X
% And so will do each returned cluster
%

    K = 1;
    clusters{K} = data;

end

