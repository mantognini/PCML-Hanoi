% Given the best learners for classification, compute rmse
clear all;
load('HaNoi_classification.mat');

% initial parameters
FinalSeed = 93726;
splitRatio = 0.7;

% define method
method = @finalMethodClassifiction;

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


%% Quick'n'Dirty adaptation of testStrategies for final method

clear all;
data = loadClassificationData();

% Settings
finalSeed = 93726;
splitRatio = 0.7;   % training-validation ratio per cluster

cluster = dummyAndNorm(data);

N = size(cluster.train.X, 1);

setSeed(finalSeed);

% Split data into training and validation sets
idx = randperm(N);
X = cluster.train.X(idx, :);
y = cluster.train.y(idx);

[XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);

% Remove outliers
[XTr, yTr] = outliersFilter(XTr, yTr);


% Collect predictions
probabilities = finalMethodClassifiction(XTr, yTr, XValid);
csvwrite('predictions_classification.csv', probabilities);

% Compute error
error = zeroOneLoss(yValid, sigmToZeroOne(probabilities));
fprintf(['0-1 Loss error is ' num2str(error) '; report it in test_errors_classification.csv\n']);


