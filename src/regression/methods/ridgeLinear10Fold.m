function yValidPred = ridgeLinear10Fold(XTr, yTr, XValid)
% ridgeLinear10Fold(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi function. The best
%   lambda is chosen by 10-Fold.
%   
    yValidPred = predictRidgeKFold(XTr, yTr, XValid, 10, @polynomialPhi, 1);
end
