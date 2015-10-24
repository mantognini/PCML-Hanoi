function yValidPred = ridgeEmplified10Fold(XTr, yTr, XValid)
% ridgeEmplified10Fold(XTr, yTr, XValid)
%   Predict using ridge regression with emplified phi function. The best
%   lambda is chosen by 10-Fold.
%   
    yValidPred = predictRidgeKFold(XTr, yTr, XValid, 10, @emplifyFeatures, -1);
end
