function yValidPred = clusterMeansMethod(XTr, yTr, XValid)
% clusterMeansMethod(XTr, yTr, XValid)
%   Classify input in three sources and predict mean of sources output.
%

    % Split into clusters
    splitTr = manualSplit(XTr);

    % Compute means
    K = 3;
    meanY = zeros(K, 1);
    for k = 1:K
        meanY(k) = mean(yTr(splitTr.idx{k}));
    end

    % predict outputs for validation set
    splitValid = manualSplit(XValid);
    yValidPred = zeros(length(XValid), 1);
    for k = 1:K
        yValidPred(splitValid.idx{k}) = meanY(k);
    end
end

