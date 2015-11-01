function yValidPred = PLRLSNewtonMethod(XTr, yTr, XValid)
%
    % Compute model
    NTr = length(yTr);
    tXTr = [ones(NTr, 1) XTr];
    K = 5;
    lambda = bestLambdaPenLog(yTr, tXTr, K);
    fprintf(['PLRLS-Newton lambda: ' num2str(lambda) ', K = ' num2str(K) '\n']);
    beta = PLRLSNewton(yTr, tXTr, lambda);
    
    % Predict
    NValid = size(XValid, 1);
    txValid = [ones(NValid, 1) XValid];
    yValidPred = sigmToZeroOne(sigmoid(txValid * beta));
end

