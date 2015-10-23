function yValidPred = clusterGDGroupA(XTr, yTr, XValid)
% clusterGradientDescentGroupA(XTr, yTr, XValid)
%   Classify input in three sources and predict output
%   by using the gradient descent method on the training data.
%
%   With this method we ignore features from group B (discrete features) to
%   keep only features from group A (continuous features)
%

    % Index of discrete features
    groupB = [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61];

    % Split into clusters
    splitTr = manualSplit(XTr);
    splitValid = manualSplit(XValid);
        
    % Remove feature from group B
    % NB: manualSplit needs all features to work so we don't remove those
    % before
    XTr(:, groupB) = [];
    XValid(:, groupB) = [];

    % Compute gradient descent
    K = 3;
    D = size(XTr, 2);
    beta = zeros(D + 1, K); % beta from least squares for each cluster
    for k = 1:K
        % .. for current cluster
        X = normalize(XTr(splitTr.idx{k}, :));
        tX = [ones(size(X, 1), 1) X];
        y = yTr(splitTr.idx{k}, :);
        beta(:, k) = leastSquaresGD(y, tX, 0.05);
    end

    % predict outputs for validation set
    yValidPred = zeros(length(XValid), 1);
    for k = 1:K
        X = XValid(splitValid.idx{k}, :);
        tX = [ones(size(X, 1), 1) X];
        yValidPred(splitValid.idx{k}) = tX * beta(:, k);
    end
end

