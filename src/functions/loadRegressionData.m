function [ data ] = loadRegressionData()
%LOADREGRESSIONDATA() Load data for repression analysis
%
% The returned data is presented as a structure with the following
% interesting fields inside
%   - dirty.
%       - train.X
%       - train.y
%       - train.Xnorm
%       - test.X
%       - test.Xnorm
%   - clean.
%       - same as above, but with the outliers removed
% which all have three cells named 1, 2 and 3 that correspond to the three
% input sources according to a manual heuristic split on input features
% 25 and 62.
%

    % Load data from file
    load('HaNoi_regression.mat');
    
    % Apply manual splitting of the identified three input sources
    X25_train = X_train(:, 25);
    X62_train = X_train(:, 62);
    X25_test  = X_test(:, 25);
    X62_test  = X_test(:, 62);

    lim62 = 15.75;
    lim25 = 15.25;
    
    idx62_train = X62_train >= lim62;
    idx62_test  = X62_test  >= lim62;
    
    idx25_train = X25_train < lim25;
    idx25_test  = X25_test  < lim25;
    
    % Indexes range over 1, 2 and 3
    idx_train = idx62_train + (idx25_train & idx62_train) + 1;
    idx_test  = idx62_test  + (idx25_test  & idx62_test)  + 1;

    % Split training data & group them into a vector of cells (which are matrices)
    for k = 1:3
        data.dirty.train.X{k} = X_train(idx_train == k, :);
        data.dirty.test.X{k}  = X_test(idx_test == k, :);
        
        data.dirty.train.y{k} = y_train(idx_train == k, :);
        
        % Normalize data as well
        data.dirty.train.Xnorm{k} = normalize(data.dirty.train.X{k});
        data.dirty.test.Xnorm{k}  = normalize(data.dirty.test.X{k});
    end
    
    % Create datasets without outliers
    % Copy dirty dataset
    data.clean = data.dirty;

    % We need to handle discrete features differently (or not at all?)
    groupB = [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61]; % Discrete features
    D = size(data.dirty.test.X{1}, 2);
    K = 3;

    % Filter out outliers
    isInit = 0;
    for f = 1:D % For each feature
        if all(~ismember(groupB, f))
            for k = 1:K
                X = data.dirty.train.Xnorm{k};
                feature = X(:, f);
                stddev(f, k) = std(feature);
                idx = abs(feature) >= 3 * stddev(f, k); % 99.7%

                % Combine outliers indices
                if isInit
                    idx_outliers{k} = idx_outliers{k} | idx;
                else
                    idx_outliers{k} = idx;
                end
            end
            isInit = 1;
        end
    end

    % Remove outliers
    for k = 1:K
        data.clean.train.Xnorm{k}(idx_outliers{k}, :) = [];
        data.clean.train.X{k}(idx_outliers{k}, :) = [];
        data.clean.train.y{k}(idx_outliers{k}, :) = [];
    end
end

