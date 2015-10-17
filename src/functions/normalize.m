function [X] = normalize(X)
% NORMALIZE(X)
%   normalize matrix X
%

    meanX = mean(X); % row vector of mean of column of X
    meanX = repmat(meanX, size(X, 1), 1); % replicate row vector
    X = X - meanX;
    stdX = std(X); % row vector of std of column of X
    stdX = repmat(stdX, size(X, 1), 1); % replicate row vector
    X = X ./ stdX;
    
end
