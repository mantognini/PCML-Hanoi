% Test various methods for regression dataset
clear all;

% Load dataset
allData = loadRegressionData();
data = allData.original;

% Settings
seeds = 10;         % number of seed to be tested
splitRatio = 0.7;   % training-validation ratio per cluster

% Data splitting methods
splitters = {
    @noClusterSplitter,
    @manualClusterSplitter,
};

% Model methods
methods = {
    @constantMethod,
    @medianMethod,
    @meanMethod,
    @GDLSMethod,
};

% Compute & plot RMSE for each data splitting & model strategies

% For each splitting strategy
for splitterNo = 1:numel(splitters)
    splitter = splitters{splitterNo};
    
    [K, clusters] = splitter(data);
    
    % Test all methods on each clusters
    for k = 1:K
        cluster = clusters{k};
        
        rmse = zeros(seeds, numel(methods));
        
        N = size(cluster.train.X, 1);
    
        % Test each methods several times with different training and 
        % validation split of the data
        for seed = 1:seeds
            setSeed(seed);
            
            % Split data into training and validation sets
            idx = randperm(N);
            X = cluster.train.X(idx, :);
            y = cluster.train.y(idx);
            
            [XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);
            
            % Test each method
            for methodNo = 1:numel(methods)
                method = methods{methodNo};
                
                % Collect predictions
                yValidPred = method(XTr, yTr, XValid);
                
                % Compute error
                rmse(seed, methodNo) = computeRmse(yValidPred - yValid);
            end % methods

        end % seeds
        
        % Plot RMSE for this cluster
        figure('Name', ['RMSE for ' func2str(splitter) ' and cluster ' num2str(k)]);
        boxplot(rmse, 'labels', cellfun(@func2str, methods, 'UniformOutput', false));
        title([num2str(k) 'th cluster']);
        xlabel('methods');
        ylabel('RMSE');
        
    end % clusters
    
end % splitter

