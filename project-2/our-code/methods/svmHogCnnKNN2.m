function yPred = svmHogCnnKNN2(train, XValid)
%
% Apply svm to both hog and cnn with maximum confidence (MC) voting

    % Make y binary
    train.y = toBinary(train.y);
    
    % Split data into (Tr, Te)
    ratio = 0.9;
    N = size(train.y, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTr = idx(1:splitIdx);
    idxTe = idx(splitIdx + 1:end);
    
    XTr.cnn = train.X.cnn(idxTr, :);
    XTr.hog = train.X.hog(idxTr, :);
    XTe.cnn = train.X.cnn(idxTe, :);
    XTe.hog = train.X.hog(idxTe, :);
    
    yTr = train.y(idxTr);
    yTe = train.y(idxTe);
    
    % Cnn on (Te, Valid)
    M = 150;
    C = 3.25;
    gamma = 0.00023;
    yPredCnn = svmPca(XTr.cnn, yTr, [XTe.cnn; XValid.cnn], M, @rbfKernel, C, gamma);
    
    % Hog on (Te, Valid)
    C = 2;
    gamma = 0.00023;
    yPredHog = svmF(XTr.hog, yTr, [XTe.hog; XValid.hog], @rbfKernel, C, gamma);
    
    % Split predictions of (Te, Valid)
    NTe = length(idxTe);
    yTePredCnn = yPredCnn(1:NTe);
    yValidPredCnn = yPredCnn(NTe + 1:end);
    yTePredHog = yPredHog(1:NTe);
    yValidPredHog = yPredHog(NTe + 1:end);
    
    % Build knn model on Te
    X1 = [yTePredCnn yTePredHog];
    knnModel = fitcknn(X1, yTe, 'NumNeighbors', 5, 'Standardize', 1);
    
    % Predict with the model Valid
    X2 = [yValidPredCnn yValidPredHog];
    yPred = predict(knnModel, X2);
    
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
