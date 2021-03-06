function yPred = logRegHog2(tr, XValid, category)
%
% Logistic Regression on pca'd hog

    % Make y binary
    tr.y = toBinary(tr.y, category);

    % Apply pca
    [X2, XValid2] = pcaManual(tr.X.hog, XValid.hog, 15);
    
    % Normalize (we apply GD!)
    [X3, ~, ~] = zscore(X2);
    [XValid3, ~, ~] = zscore(XValid2);
    
    % Form tX
    tX3 = [ones(size(X3, 1), 1) X3];
    tXValid3 = [ones(size(XValid3, 1), 1) XValid3];
    
    % Find good lambda (0.27 seems good)
    %lambda = findLambdaPLR(tr.y, tX3, 10);
    lambda = 0.27;
    
    % Find model
    beta = PLRLS(tr.y, tX3, lambda);
    
    % Predict
    yPred = sigmToZeroOne(sigmoid(tXValid3 * beta));
end