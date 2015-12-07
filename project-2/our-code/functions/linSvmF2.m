function yPred = linSvmF2(X, y, XValid, C)
    % Normalize data
    X = normalize(X);
    XValid = normalize(XValid);
    
    % Binary train y {-1, 1}
    otherIdx = (y == 0);
    y(otherIdx) = -1;
    y(~otherIdx) = 1;
    
    % Find alpha star
    KTrain = linearKernel(X, X);
    [alpha, beta0] = SMO(KTrain, y, C);
    indexSV = (alpha > 0);
    
    % Predict
    KValid = linearKernel(XValid, X(indexSV, :));
    yPred = KValid * (alpha(indexSV) .* y(indexSV)) + beta0;
    
    % Final decision
    otherIdx = (yPred < 0);
    yPred(otherIdx) = 0;
    yPred(~otherIdx) = 1;
end