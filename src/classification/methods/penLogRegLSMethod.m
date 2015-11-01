function yValidPred = penLogRefLSMethod(XTr, yTr, XValid)
%
    % Compute model
    NTr = length(yTr);
    tXTr = [ones(NTr, 1) XTr];
    K = 10;
    lambda = bestLambdaPenLog(yTr, tXTr, K);
    fprintf(['penLog lambda: ' num2str(lambda) ', K = ' num2str(K) '\n']);
    beta = penLogRegLS(yTr, tXTr, lambda);
    
    % Predict
    NValid = size(XValid, 1);
    txValid = [ones(NValid, 1) XValid];
    yValidPred = sigmToZeroOne(sigmoid(txValid * beta));
end

