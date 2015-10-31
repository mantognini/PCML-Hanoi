function yValidPred = ridgeSquare10Fold(XTr, yTr, XValid)
% ridgeSquare10Fold(XTr, yTr, XValid)
%   Predict using ridge regression with square phi function. The best
%   lambda is chosen by 10-Fold.
%   
    [~, yValidPred] = predictRidgeKFold(XTr, yTr, XValid, 10, @polynomialPhi, 2);
end
