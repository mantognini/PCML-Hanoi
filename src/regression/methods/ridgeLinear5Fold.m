function yValidPred = ridgeLinear5Fold(XTr, yTr, XValid)
% ridgeLinear5Fold(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi function. The best
%   lambda is chosen by 5-Fold.
%   
    yValidPred = predictRidgeKFold(XTr, yTr, XValid, 5, @polynomialPhi, 1);
end
