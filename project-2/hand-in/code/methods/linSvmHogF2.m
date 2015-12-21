function yPred = linSvmHogF2(train, XValid, category, C)
%
% Linear svm on the hog feature using a fixed C
    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Predict
    yPred = linSvmF2(train.X.hog, train.y, XValid.hog, C);
end