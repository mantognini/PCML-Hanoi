function yValidPred = logisticClassificationWithManualSplittingMethod(XTr, yTr, XValid)
%logisticClassificationMethod Apply logistic regression on data for
%classification except with the 11th feature is > -10.
%

    % Remove X(:, 11) > -10 from training data
    idx = XValid(:, 11) > -10;
    XTr(idx, :) = [];
    yTr(idx) = [];

    % Estimate the model
    N = length(yTr);
    tXTr = [ones(N, 1) XTr];
    alpha = 0.1;
    beta = logisticRegression(yTr, tXTr, alpha);
    
    % Predict classification for validation set
    N = size(XValid, 1);
    tXValid = [ones(N, 1) XValid];
    yValidPred = tXValid * beta;
    
    %histogram(yValidPred) % for debugging if needed
    
    % Use sigmoid with 50% threshold for class mapping
    sigma = @(x) exp(x) ./ (1 + exp(x));
    theta = 0.5;
    yValidPred = sigma(yValidPred);
    yValidPred(yValidPred >= theta) = 1;
    yValidPred(yValidPred <  theta) = 0;
    
    % Override X(:, 11) > -10 to 1 in validation set
    yValidPred(XValid(:, 11) > -10) = 1;

end

