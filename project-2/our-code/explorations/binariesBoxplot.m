%
% Save final predictions for the binary task
clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
addpath(genpath('toolboxs/'));
load 'data/data.mat';

nbRuns = 20;
ratio = 0.7;
errors = zeros(nbRuns, 4);

for r = 1:nbRuns
    fprintf(['run ' num2str(r) '/' num2str(nbRuns) '\n']);
    
    % Split the data
    N = size(data.yTrain, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    tr.X.hog = data.hog.train.X(idxTrain, :);
    tr.X.cnn = data.cnn.train.X(idxTrain, :);
    tr.cnn.mu = data.cnn.mu;
    tr.hog.mu = data.hog.mu;
    tr.cnn.sigma = data.cnn.sigma;
    tr.hog.sigma = data.hog.sigma;
    tr.y = data.yTrain(idxTrain); % train y are 4-class

    val.X.hog = data.hog.train.X(idxValid, :);
    val.X.cnn = data.cnn.train.X(idxValid, :);
    val.y = data.yTrain(idxValid); % valid y are 4-class

    % apply methods
    M = 150;
    [TrZ, VaZ] = pcaManual(tr.X.cnn, val.X.cnn, M);

    try1 = toBinary(tr.y, 1);
    try2 = toBinary(tr.y, 2);
    try3 = toBinary(tr.y, 3);
    try4 = toBinary(tr.y, 4);

    genericClassifier = @(y, C, gamma) svmF(TrZ, y, VaZ, @rbfKernel, C, gamma);

    % C and gamma where empirically found
    yPred1 = genericClassifier(try1, 7, 0.0003);
    yPred2 = genericClassifier(try2, 1, 3.5e-4);
    yPred3 = genericClassifier(try3, 10, 1e-4);
    yPred4 = genericClassifier(try4, 3.25, 0.00023);

    otherIdx = (yPred1 < 0);
    yPred1(otherIdx)  = 0;
    yPred1(~otherIdx) = 1;

    otherIdx = (yPred2 < 0);
    yPred2(otherIdx)  = 0;
    yPred2(~otherIdx) = 1;

    otherIdx = (yPred3 < 0);
    yPred3(otherIdx)  = 0;
    yPred3(~otherIdx) = 1;

    otherIdx = (yPred4 < 0);
    yPred4(otherIdx)  = 0;
    yPred4(~otherIdx) = 1;
    
    errors(r, 1) = BER(yPred1, toBinary(val.y, 1));
    errors(r, 2) = BER(yPred2, toBinary(val.y, 2));
    errors(r, 3) = BER(yPred3, toBinary(val.y, 3));
    errors(r, 4) = BER(yPred4, toBinary(val.y, 4));
end

%%
names{1} = 'plane';
names{2} = 'car';
names{3} = 'horse';
names{4} = 'other';


figure('Name', 'BER BINARY');
boxplot(errors, 'labels', names);
title(['BER for binary SVM on each label']);
