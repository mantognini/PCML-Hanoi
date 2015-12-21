function yPred = svmHogCnnForest2(train, XValid, category)
%
% Apply svm to both hog and cnn and forest decision

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Define methods
    m1 = @(X, y, XValid) svmF(X.hog, y, XValid.hog, @rbfKernel, 2, 0.00023);
    m2 = @(X, y, XValid) svmPca(X.cnn, y, XValid.cnn, 150, @rbfKernel, 3.25, 0.00023);
    
    % Predict
    ops.M = 150;
    ops.F1 = 2;
    f = @(a, b, c) forestComb2(a, b, c, ops);
    yPred = multiPred(train.X, train.y, XValid, 1, 0.7, {m1, m2}, f);
    
end
