% Test various methods for regression dataset
clear all;

% load dataset
allData = loadRegressionData();

% initialization
methods = {
    %@constantMethod,
    %@medianMethod,
    %@meanMethod,
    %@clusterMediansMethod,
    %@clusterMeansMethod,
    @clusterGD,
    @clusterGDGroupA,
    @clusterGDLS,
    @clusterGDLSGroupA,
    %@clusterLeastSquares,
};

seeds = 10;
splitRatio = 0.7;

N = size(allData.original.train.X, 1);

% Compute expected error estimate (independent of training dataset)
for seed = 1:seeds
    setSeed(seed);
    
    % Split data into training and validation sets
    idx = randperm(N);
    X = allData.original.train.X(idx, :);
    y = allData.original.train.y(idx);
    
    [XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);
    
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

