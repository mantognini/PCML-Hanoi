% Given the best learners for classification, compute rmse
load('HaNoi_classification.mat');

% initial parameters
FinalSeed = 93726;
splitRatio = 0.7;

% define method
method = @finalMethod;

% A final seed
setSeed(FinalSeed);

% Split data into training and validation sets
N = size(X_train, 1);
idx = randperm(N);
X = X_train(idx, :);
y = y_train(idx);
[XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);

% Collect predictions
[yValidPred, yTestPred, testRMSE] = method(XTr, yTr, XValid, X_test);

% Compute 0-1 loss
% todo

% save yTestPred
csvwrite('predictions_classification.csv', yTestPred);