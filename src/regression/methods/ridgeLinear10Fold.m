function yValidPred = ridgeLinear10Fold(XTr, yTr, XValid)
% ridgeLinear(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi functions. The best
%   lambda is chosen by K-Fold.
%   

    % Form new components space
    N = length(yTr);
    tXTr = [ones(N, 1) XTr];
    
    lambda = bestLambdaKFold(yTr, tXTr, 10);
    beta = ridgeRegression(yTr,tXTr, lambda);
    
    NValid = size(XValid, 1);
    tXValid = [ones(NValid, 1) XValid];
    yValidPred = tXValid*beta;
end
