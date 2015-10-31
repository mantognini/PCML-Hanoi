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

% todo: plot
%%
figure('Name', 'Learning curve');
boxplot(validRMSE, 'color', 'b', 'plotstyle', 'compact', 'labels', ratios);
% legend(findobj(gca,'Tag','Box'), 'valid');
%legend('valid');
hold on;
boxplot(trainRMSE, 'color', 'r', 'plotstyle', 'compact', 'labels', ratios);
% legend(findobj(gca,'Tag','Box'), 'train');
% legend(findobj(gca,'Tag','Box'), 'train', 'valid');
xlabel('Training set size');
ylabel('RMSE');
ylim([200 800]);

