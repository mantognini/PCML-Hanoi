function yValidPred = clusterMeansMethod(XTr, yTr, XValid)
% clusterMeansMethod(XTr, yTr, XValid)
%   Classify input in three sources and predict mean of sources output.
%

    % todo: to refactor..

    % split the data in training/validation sets
    setSeed(1);
    N = size(data.original.train.X, 1);
    idx = randperm(N);
    X = data.original.train.X(idx, :);
    y = data.original.train.y(idx);

    [XTr, yTr, XValid, yValid] = split(y, X, 0.7);

    % build naive model on training set
    splitTr = manualSplit(XTr, yTr);

    K = 3;
    for k = 1:K
        meanY(k) = mean(splitTr.y{k});
    end

    % predict outputs for validation set
    splitValid = manualSplit(XValid, yValid);
    pred = zeros(length(XValid), 2);
    for k = 1:K
        subIdx = splitValid.idxNo{k};

        % The prediction is the group train output mean
        pred(subIdx, 1) = ones(length(subIdx), 1) * meanY(k);

        % And this is the correct answer
        pred(subIdx, 2) = splitValid.y{k};
    end

    % prediction error
    diff = abs(pred(:, 1) - pred(:, 2));
    error = 2*sqrt(diff'*diff);
end

