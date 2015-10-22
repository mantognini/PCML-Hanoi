function [ data ] = loadRegressionData()
%LOADREGRESSIONDATA() Load data for repression analysis
%
% The returned data is presented as a structure with the following
% interesting fields:
%   - train.X
%   - train.y
%   - train.Xnorm
%   - test.X
%   - test.Xnorm
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
        data.train.X{k} = X_train(idx_train == k, :);
        data.test.X{k}  = X_test(idx_test == k, :);
        
        data.train.y{k} = y_train(idx_train == k, :);
        
        % Normalize data as well
        data.train.Xnorm{k} = normalize(data.train.X{k});
        data.test.Xnorm{k}  = normalize(data.test.X{k});
    end
    
end

