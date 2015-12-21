function yPred = svmHogCnnMC2(train, XValid, category)
%
% Apply svm to both hog and cnn with maximum confidence (MC) voting

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Cnn
    M = 150;
    C = 3.25;
    gamma = 0.00023;
    yPredCnn = svmPca(train.X.cnn, train.y, XValid.cnn, M, @rbfKernel, C, gamma);
    
    % Hog
    C = 2;
    gamma = 0.00023;
    yPredHog = svmF(train.X.hog, train.y, XValid.hog, @rbfKernel, C, gamma);
    
    % Maximum confidence voting
    cnnIdx = (abs(yPredCnn) > abs(yPredHog));
    yPred = yPredHog;
    yPred(cnnIdx) = yPredCnn(cnnIdx);
    
    % Binary decision
    otherIdx = (yPred < 0);
    yPred(otherIdx) = 0;
    yPred(~otherIdx) = 1;
    
%     % Optionally
%     % Plot
%     figure('Name', 'Svm {Hog, Cnn}');
%     plot(yPredHog(cnnIdx), yPredCnn(cnnIdx), 'b.');
%     hold on;
%     plot(yPredHog(~cnnIdx), yPredCnn(~cnnIdx), 'r.');
%     hold on;
%     legend('blue = cnn voted', 'Location', 'best');
%     xlabel('hog');
%     ylabel('cnn');
%     hold off;
end
