% Given the best learners for regression, compute rmse
clear all;
load('HaNoi_regression.mat');

% initial parameters
FinalSeed = 59283;
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

% Compute rmse
e = computeRmse(yValidPred - yValid);
fprintf(['Final rmse is ' num2str(e) '; write it into test_errors_regression.csv\n']);

% save yTestPred
csvwrite('predictions_regression.csv', yTestPred);

