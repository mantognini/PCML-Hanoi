function yPred = linSvmPcaCnnF2(train, XValid, C, M)
%
% Linear svm on pca'd cnn feature with fixed C and M

    % Make y binary
    train.y = toBinary(train.y);
    
    % Predict
    yPred = svmPca2(train.X.cnn, train.y, XValid.cnn, M, @linearKernel, C, []);
end
