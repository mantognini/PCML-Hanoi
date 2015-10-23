function yValidPred = clusterMediansMethod(XTr, yTr, XValid)
% clusterMediansMethod(XTr, yTr, XValid)
%   Classify input in three sources and predict median of sources output.
%

    % Split into clusters
    splitTr = manualSplit(XTr);

    % Compute medians
    K = 3;
    medianY = zeros(K, 1);
    for k = 1:K
        medianY(k) = median(yTr(splitTr.idx{k}));
    end

    % predict outputs for validation set
    splitValid = manualSplit(XValid);
    yValidPred = zeros(length(XValid), 1);
    for k = 1:K
        yValidPred(splitValid.idx{k}) = medianY(k);
    end
end

