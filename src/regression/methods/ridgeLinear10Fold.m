function yValidPred = ridgeLinear10Fold(XTr, yTr, XValid)
% ridgeLinear(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi functions. The best
%   lambda is chosen by 10-Fold.
%   
    yValidPred = predictRidgeLinearKFold(XTr, yTr, XValid, 10);
end
