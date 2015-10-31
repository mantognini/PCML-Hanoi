function yValidPred = ridgeSqrt10Fold(XTr, yTr, XValid)
% ridgeSqrt10Fold(XTr, yTr, XValid)
%   Predict using ridge regression with sqrt phi function. The best
%   lambda is chosen by 10-Fold.
%   
    [~, yValidPred] = predictRidgeKFold(XTr, yTr, XValid, 10, @polynomialPhi, 0.5);
end
