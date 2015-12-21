function yPred = rbfSvmHogF2(train, XValid, category, C, gamma)
%
% Rbf svm on the hog feature using a fixed C, gamma
    % Make y binary
    train.y = toBinary(train.y, category);
    
    
    % Pca on hog
    [train.X.hog, XValid.hog] = pcaManual(train.X.hog, XValid.hog, 15);
    
    % Predict
    yPred = svmF2(train.X.hog, train.y, XValid.hog, @rbfKernel, C, gamma);
end