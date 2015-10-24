function yValidPred = predictRidgeLinearKFold(XTr, yTr, XValid, K)
% predictRidgeLinearKFold(XTr, yTr, XValid)
%   Predict using ridge regression with linear phi functions. The best
%   lambda is chosen by K-Fold.
%   It controls that K <= N
%   

    % Form new components space
    N = length(yTr);
    tXTr = [ones(N, 1) XTr];
    
    lambda = bestLambdaKFold(yTr, tXTr, min(K, N));
    beta = ridgeRegression(yTr,tXTr, lambda);
    
    NValid = size(XValid, 1);
    tXValid = [ones(NValid, 1) XValid];
    yValidPred = tXValid*beta;
end
