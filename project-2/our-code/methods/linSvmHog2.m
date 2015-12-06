function yPred = linSvmHog2(train, XValid)
    % Normalize data
    train.X.hog = normalize(train.X.hog);
    XValid.hog = normalize(XValid.hog);
    
    % Binary train y {-1, 1}
    otherIdx = (train.y == 4);
    train.y(otherIdx) = -1;
    train.y(~otherIdx) = 1;
    
    % Find alpha star
    C = 1;
    KTrain = linearKernel(train.X.hog, train.X.hog);
    [alpha, beta0] = SMO(KTrain, train.y, C);
    indexSV = (alpha > 0);
    
    % Predict
    KValid = linearKernel(XValid.hog, train.X.hog(indexSV, :));
    yPred = KValid * (alpha(indexSV) .* train.y(indexSV)) + beta0;
    
    % Final decision
    otherIdx = (yPred < 0);
    yPred(otherIdx) = 0;
    yPred(~otherIdx) = 1;
end