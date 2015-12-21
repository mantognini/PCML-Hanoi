function yPred = svmHogCnnCustom2(train, XValid)
%
% Apply svm to both hog and cnn with a custom voting

    % Make y binary
    train.y = toBinary(train.y);
    
    % Cnn
    M = 150;
    C = 3.25;
    gamma = 0.00023;
    yPredCnn = svmPca(train.X.cnn, train.y, XValid.cnn, M, @rbfKernel, C, gamma);
    
    % Hog
    C = 2;
    gamma = 0.00023;
    yPredHog = svmF(train.X.hog, train.y, XValid.hog, @rbfKernel, C, gamma);
    
    % Cnn decides
    yPred = yPredCnn;
    
    % .. unless cnn is not sure and but hog is
    cIdx = (abs(yPredCnn) < 1 & abs(yPredHog) > 1);
    yPred(cIdx) = yPredHog(cIdx);
    
    % Binary decision
    otherIdx = (yPred < 0);
    yPred(otherIdx) = 0;
    yPred(~otherIdx) = 1;
    
    % Optionally
    fprintf(['Hog saved cnn ' num2str(sum(cIdx)) ' times.\n']);
end
