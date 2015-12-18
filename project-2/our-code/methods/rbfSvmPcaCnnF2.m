function yPred = rbfSvmPcaCnnF2(train, XValid, category, M, C, gamma)
%
% Rbf svm on pca'd cnn feature with fixed C and M

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Predict
    yPred = svmPca2(train.X.cnn, train.y, XValid.cnn, M, @rbfKernel, C, gamma);
end
