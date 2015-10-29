% Given the best learners for regression, compute rmse
clear all;
load('HaNoi_regression.mat');
data.train.X = X_train;
data.train.y = y_train;
data.test.X  = X_test;

% initial parameters
S = 10;
splitRatio = 0.7;

% define method
obj = FinalMethod();
method = @obj.apply;

% A final seed
setSeed(1);

% Split data into training and validation sets
N = size(data.train.X, 1);
idx = randperm(N);
X = data.train.X(idx, :);
y = data.train.y(idx);
[XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);

% Collect predictions
[yValidPred, yTestPred] = method(XTr, yTr, XValid, data.test.X);

% Compute error for this cluster
e = computeRmse(yValidPred - yValid);
display(e, 'rmse');

% todo: save yTestPred

