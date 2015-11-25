% Given the best learners for regression, compute rmse
clear all;
load('HaNoi_regression.mat');

% initial parameters
ratios = .6:.025:0.9;
seeds = 1:10;

% define method
method = @finalMethod;

% Compute
validRMSE = zeros(length(seeds), length(ratios));
trainRMSE = zeros(length(seeds), length(ratios));
for ratio = ratios
    ratioId = find(ratios == ratio);
    for seed = seeds
        % A final seed
        setSeed(seed);

        % Split data into training and validation sets
        [XTr, yTr, XValid, yValid] = doSplit(y_train, X_train, ratio);

        % Collect predictions
        [yValidPred, ~, rmseTr] = method(XTr, yTr, XValid, X_test);

        % Compute error for this cluster
        validRMSE(seed, ratioId) = computeRmse(yValidPred - yValid);
        trainRMSE(seed, ratioId) = rmseTr;
    end
end

figure('Name', 'Learning curve');
rmse = reshape([validRMSE; trainRMSE], 2, length(seeds), length(ratios));
aboxplot(rmse, 'labels', ratios, 'colorgrad', 'orange_down');
xlabel('Training set size');
ylabel('RMSE');
legend('train', 'valid');
title(['Learning curve for ' func2str(method)]);
