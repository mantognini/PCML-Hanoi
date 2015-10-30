function yValidPred = predictRidgeKFold(XTr, yTr, XValid, K, phi, d)
% predictRidgeKFold(XTr, yTr, XValid, K, phi)
%   Predict using ridge regression with phi function applied to each feature
%   giving it the d parameter (d = 1(linear?), 2(square), etc..).
%   Note that d might not be considered by phi.
%   The best lambda is chosen by K-Fold.
%   It controls that K <= N
%   

    % Form new components space
    N = length(yTr);
    tXTr = [ones(N, 1) phi(XTr, d)];
    
    lambda = bestLambdaKFold(yTr, tXTr, min(K, N));
    beta = ridgeRegression(yTr, tXTr, lambda);
    %disp(beta);
    
    NValid = size(XValid, 1);
    tXValid = [ones(NValid, 1) phi(XValid, d)];
    yValidPred = tXValid * beta;
end
