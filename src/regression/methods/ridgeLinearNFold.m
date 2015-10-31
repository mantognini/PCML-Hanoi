function yValidPred = ridgeLinearNFold(XTr, yTr, XValid)
% ridgeLinearNFold(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi function. The best
%   lambda is chosen by N-Fold.
%   
    [~, yValidPred] = predictRidgeKFold(XTr, yTr, XValid, length(yTr), @polynomialPhi, 1);
end
