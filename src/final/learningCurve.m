% Given the best learners for regression, compute rmse
clear all;
load('HaNoi_regression.mat');

% initial parameters
ratios = .4:.1:0.9;
seeds = 1:10;

% define method
obj = FinalMethod();
method = @obj.apply;

% Compute
validRMSE = zeros(length(seeds), length(ratios));
for ratio = ratios
    ratioId = find(ratios == ratio);
    for seed = seeds
        % A final seed
        setSeed(seed);

        % Split data into training and validation sets
        N = size(X_train, 1);
        idx = randperm(N);
        X = X_train(idx, :);
        y = y_train(idx);
        [XTr, yTr, XValid, yValid] = doSplit(y, X, ratio);

        % Collect predictions
        [yValidPred, ~, testRMSE] = method(XTr, yTr, XValid, X_test);

        % Compute error for this cluster
        validRMSE(seed, ratioId) = computeRmse(yValidPred - yValid);
        testRMSE(seed, ratioId) = testRMSE;
    end
end

% todo: plot
figure('Learning curve');
boxplot(validRMSE, ratios, 'colors', 'r');
hold on;
boxplot(validRMSE, ratios, 'colors', 'b');
hold off;
xlabel('Training set size');
ylabel('RMSE');

