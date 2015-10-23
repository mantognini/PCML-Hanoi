% Test various methods for regression dataset
clear all;

% load dataset
allData = loadRegressionData();

% initialization
% Methods not plotted:
% @constantMethod, @meanMethod, @clusterLeastSquares @clusterMeansMethod
methods = {...
    @clusterGradientDescent}; % 
seeds = 2;
prop = 0.95;

N = size(allData.original.train.X, 1);

% Compute expected error estimate (independent of training dataset)
for seed = 1:seeds
    setSeed(seed);
    
    % Split data into training and validation sets
    idx = randperm(N);
    X = allData.original.train.X(idx, :);
    y = allData.original.train.y(idx);
    
    [XTr, yTr, XValid, yValid] = doSplit(y, X, prop);
    
    % Test each method
    for methodNo = 1:numel(methods)
        method = methods{methodNo};
        
        % Collect predictions
        yValidPred = method(XTr, yTr, XValid);
        
        % Compute error
        rmse(seed, methodNo) = computeRmse(yValidPred - yValid);
    end
end

% Plot methods
fig = figure();

boxplot(rmse, 'labels', cellfun(@func2str, methods, 'UniformOutput', false));
