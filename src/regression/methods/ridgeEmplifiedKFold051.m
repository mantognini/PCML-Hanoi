function yValidPred = ridgeEmplifiedKFold051(K, XTr, yTr, XValid)
% ridgeEmplifiedKFold(K, XTr, yTr, XValid)
%   Predict using ridge regression with emplified phi function. The best
%   lambda is chosen by K-Fold.
%   
    yValidPred = predictRidgeKFold(XTr, yTr, XValid, K, @emplifyFeatures051, -1);
end
