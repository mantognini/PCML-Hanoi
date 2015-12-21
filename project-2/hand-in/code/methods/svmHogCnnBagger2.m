function yPred = svmHogCnnBagger2(train, XValid)
%
% Apply svm to both hog and cnn and forest decision

    % Make y binary
    train.y = toBinary(train.y);
    
    % Define methods
    m1 = @(X, y, XValid) svmF(X.hog, y, XValid.hog, @rbfKernel, 2, 0.00023);
    m2 = @(X, y, XValid) svmPca(X.cnn, y, XValid.cnn, 150, @rbfKernel, 3.25, 0.00023);
    
    % Predict
    combF = @(X, y, XValid) treeBaggerComb(X, y, XValid, 50);
    yPred = multiPred(train.X, train.y, XValid, 10, 0.5, {m1, m2}, combF);
    
end
