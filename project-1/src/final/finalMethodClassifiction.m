function [yVaPred, yTePred] = finalMethodClassifiction(XTr, yTr, XValid, XTe)
% Based on PLRLSMethod; return p(y = 1|XTr, yTr) for XValid
    % Compute model
    NTr = length(yTr);
    tXTr = [ones(NTr, 1) XTr];
    K = 5;
    lambda = bestLambdaPenLog(yTr, tXTr, K);
    fprintf(['PLRLS lambda: ' num2str(lambda) ', K = ' num2str(K) '\n']);
    beta = PLRLS(yTr, tXTr, lambda);
    
    % Predict
    NValid = size(XValid, 1);
    NTe = size(XTe, 1);
    
    txValid = [ones(NValid, 1) XValid];
    tXTe = [ones(NTe, 1) XTe];
    
    yVaPred = sigmoid(txValid * beta);
    yTePred = sigmoid(tXTe * beta);
end

