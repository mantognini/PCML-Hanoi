function yPred = rbfSvmPcaCnnManualTree4(train, XValid, threshold)
%
% Apply SVM for binary prediction for detecting
%  - {1} or {2, 3, 4}
%  - {2} or {1, 2, 4}
%  - {3} or {1, 2, 4}
% then apply manual binary tree classification for multiclass prediction
    
    M = 150;
    [TrZ, VaZ] = pcaManual(train.X.cnn, XValid.cnn, M);
    
    try1 = toBinary(train.y, 1);
    try2 = toBinary(train.y, 2);
    try3 = toBinary(train.y, 3);
    
    genericClassifier = @(y, C, gamma) svmF(TrZ, y, VaZ, @rbfKernel, C, gamma);
    
    % C and gamma where empirically found
    yScore1 = genericClassifier(try1, 7, 0.0003);
    yScore2 = genericClassifier(try2, 1, 3.5e-4);
    yScore3 = genericClassifier(try3, 10, 1e-4);
    
    % Apply manual tree
    yPred = ones(size(XValid.cnn, 1), 1) * 4; % default: 4
    yPred(yScore3 <= threshold) = 3;
    yPred(yScore1 <= threshold) = 1; 
    yPred(yScore2 <= threshold) = 2;
    
end
