function yValidPred = ridgeLinear20Fold(XTr, yTr, XValid)
% ridgeLinear20Fold(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi function. The best
%   lambda is chosen by 20-Fold.
%   
    yValidPred = predictRidgeKFold(XTr, yTr, XValid, 20, @polynomialPhi, 1);
end
