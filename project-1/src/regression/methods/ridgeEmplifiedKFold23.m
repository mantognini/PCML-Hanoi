function yValidPred = ridgeEmplifiedKFold23(K, XTr, yTr, XValid)
% ridgeEmplifiedKFold(K, XTr, yTr, XValid)
%   Predict using ridge regression with emplified phi function. The best
%   lambda is chosen by K-Fold.
%   
    [~, yValidPred] = predictRidgeKFold(XTr, yTr, XValid, K, @emplifyFeatures23, -1);
end
