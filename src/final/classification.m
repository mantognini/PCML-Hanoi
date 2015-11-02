% Given the best learners for classification, compute rmse
% Quick'n'Dirty adaptation of testStrategies for final method

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


