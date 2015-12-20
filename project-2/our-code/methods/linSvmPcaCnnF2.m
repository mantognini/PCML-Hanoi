function yPred = linSvmPcaCnnF2(train, XValid, category, C, M)
%
% Linear svm on pca'd cnn feature with fixed C and M

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Predict
    yPred = svmPca2(train.X.cnn, train.y, XValid.cnn, M, @linearKernel, C, []);
end
