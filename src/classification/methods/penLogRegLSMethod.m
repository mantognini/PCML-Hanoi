function yValidPred = penLogRefLSMethod(XTr, yTr, XValid)
%
    % Compute model
    NTr = length(yTr);
    tXTr = [ones(NTr, 1) XTr];
    % todo compute lambda
    lambda = 0;
    beta = penLogRegLS(yTr, tXTr, lambda);
    
    % Predict
    NValid = size(XValid, 1);
    txValid = [ones(NValid, 1) XValid];
    yValidPred = sigmToZeroOne(sigmoid(txValid * beta));
end

