function yValidPred = clusterLeastSquares(XTr, yTr, XValid)
% clusterLeastSquares(XTr, yTr, XValid)
%   Classify input in three sources and predict output
%   by using the least squares method on the training data.
%

    % Split into clusters
    splitTr = manualSplit(XTr);

    % Compute least squares
    K = 3;
    D = size(XTr, 2);
    beta = zeros(D + 1, K); % beta from least squares for each cluster
    for k = 1:K
        % .. for current cluster
        X = XTr(splitTr.idx{k}, :);
        tX = [ones(size(X, 1), 1) X];
        y = yTr(splitTr.idx{k}, :);
        beta(:, k) = leastSquares(y, tX);
    end

    % predict outputs for validation set
    splitValid = manualSplit(XValid);
    yValidPred = zeros(length(XValid), 1);
    for k = 1:K
        X = XValid(splitValid.idx{k}, :);
        tX = [ones(size(X, 1), 1) X];
        yValidPred(splitValid.idx{k}) = tX * beta(:, k);
    end
end

