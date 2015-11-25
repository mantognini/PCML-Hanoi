% Given the best learners for regression, compute rmse
clear all;
load('HaNoi_classification.mat');

% initial parameters
FinalSeed = 1235;
splitRatio = 0.7;

% define method
method = @finalMethodClassifiction;

% A final seed
setSeed(FinalSeed);

% Split data into training and validation sets
y_train(y_train == -1) = 0;
N = size(X_train, 1);
idx = randperm(N);
X = X_train(idx, :);
y = y_train(idx);
[XTr, yTr, XTeSimul, yTeSimul] = doSplit(y, X, splitRatio);

% Collect predictions
[other, yTeSimulPred] = method(XTr, yTr, XTeSimul, XTeSimul);

% Compute rmse
e = zeroOneLoss(sigmToZeroOne(other), yTeSimul);
fprintf(['0-1 Loss on Test Simulation: ' num2str(e)]);

