function yValidPred = ridgeLinear5Fold(XTr, yTr, XValid)
% ridgeLinear(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi functions. The best
%   lambda is chosen by 5-Fold.
%   
    yValidPred = predictRidgeLinearKFold(XTr, yTr, XValid, 5);
end
