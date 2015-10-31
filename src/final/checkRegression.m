% Given the best learners for regression, compute rmse
clear all;
load('HaNoi_regression.mat');

% initial parameters
FinalSeed = 2341;
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
[XTr, yTr, XTeSimul, yTeSimul] = doSplit(y, X, splitRatio);

% Collect predictions
[~, yTeSimulPred, ~] = method(XTr, yTr, XTeSimul, XTeSimul);

% Compute rmse
e = computeRmse(yTeSimul - yTeSimulPred);
fprintf(['rmse on Test Simulation: ' num2str(e)]);

