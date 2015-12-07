function yPred = linSvmHogF2(train, XValid, C)
    % Make y binary
    train.y = toBinary(train.y);
    
    % Predict
    yPred = linSvmF2(train.X.hog, train.y, XValid.hog, C);
end