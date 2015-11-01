function yValidPred = LRLSMethod(XTr, yTr, XValid)
%
    % Compute model
    NTr = length(yTr);
    tXTr = [ones(NTr, 1) XTr];
    beta = LRLS(yTr, tXTr);
    
    % Predict
    NValid = size(XValid, 1);
    txValid = [ones(NValid, 1) XValid];
    yValidPred = sigmToZeroOne(sigmoid(txValid * beta));
end

