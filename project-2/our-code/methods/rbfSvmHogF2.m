function yPred = rbfSvmHogF2(train, XValid, C, gamma)
%
% Rbf svm on the hog feature using a fixed C, gamma
    % Make y binary
    train.y = toBinary(train.y);
    
    % Predict
    yPred = svmF2(train.X.hog, train.y, XValid.hog, @rbfKernel, C, gamma);
end