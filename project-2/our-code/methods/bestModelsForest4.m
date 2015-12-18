function yPred = bestModelsForest4(train, XValid)
%
% Apply best models on each class and combine them with a random forest

    % pca cnn
    [train.X.cnn, XValid.cnn] = pcaManual(train.X.cnn, XValid.cnn, 150);
    
    % Define methods
    m1 = @(X, y, XValid) binAndSvm(1, X.cnn, y, XValid.cnn, @rbfKernel, 7, 0.0003);
    m2 = @(X, y, XValid) binAndSvm(2, X.cnn, y, XValid.cnn, @rbfKernel, 1, 3.5e-4);
    m3 = @(X, y, XValid) binAndSvm(3, X.cnn, y, XValid.cnn, @rbfKernel, 10, 1e-4);
    m4 = @(X, y, XValid) binAndSvm(4, X.cnn, y, XValid.cnn, @rbfKernel, 3.25, 0.00023);
    
    % Predict
    ops.M = 150;
    ops.F1 = 4;
    f = @(a, b, c) forestComb4(a, b, c, ops);
    yPred = multiPred(train.X, train.y, XValid, 1, 0.7, {m1, m2, m3, m4}, f);
    
end

function yPred = binAndSvm(category, X, y, XValid, kernelFn, C, param)
    % Make y binary
    y = toBinary(y, category);
    
    % Apply svm
    yPred = svmF(X, y, XValid, kernelFn, C, param);
end

