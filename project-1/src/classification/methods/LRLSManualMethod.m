function yValidPred = LRLSManualMethod(XTr, yTr, XValid)
% Apply logistic regression on data for
% classification except with the 11th feature is > -10.
%

    % Remove X(:, 11) > -10 from training data
    idx = XValid(:, 11) > -10;
    XTr(idx, :) = [];
    yTr(idx) = [];

    % Compute model
    NTr = length(yTr);
    tXTr = [ones(NTr, 1) XTr];
    beta = LRLS(yTr, tXTr);
    
    % Predict
    NValid = size(XValid, 1);
    txValid = [ones(NValid, 1) XValid];
    yValidPred = sigmToZeroOne(sigmoid(txValid * beta));
    
    % Override X(:, 11) > -10 to 1 in validation set
    yValidPred(XValid(:, 11) > -10) = 1;

end

