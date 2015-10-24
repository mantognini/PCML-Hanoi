% Test various methods for regression dataset
clear all;

% Load dataset
allData = loadRegressionData();
data = allData.original;

% Settings
seeds = 10;         % number of seed to be tested
splitRatio = 0.7;   % training-validation ratio per cluster

strategies = {
    {
        @noClusterSplitter,
        @noFilter,
        % method for the unique cluster
        {
            @constantMethod,
            @medianMethod,
            @meanMethod,
            @GDLSMethod,
            @ridgeLinear10Fold,
        }
    },
    {
        @manualClusterSplitter,
        @noFilter,
        % method for the 1st cluster
        {
            %@constantMethod,
            @medianMethod,
            @meanMethod,
            @GDLSMethod,
            @ridgeLinear10Fold,
        }
        % method for the 2nd cluster
        {
            %@constantMethod,
            @medianMethod,
            @meanMethod,
            @GDLSMethod,
            @ridgeLinear10Fold,
        }
        % method for the 3rd cluster
        {
            @constantMethod,
            @medianMethod,
            @meanMethod,
            @GDLSMethod,
            @ridgeLinear10Fold,
        }
    },
};

% Compute & plot RMSE for each data splitting & model strategies

% For each splitting strategy
for splitterNo = 1:numel(strategies)
    splitter = strategies{splitterNo}{1};
    filter   = strategies{splitterNo}{2};
    methods  = strategies{splitterNo}{3};
    
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
            
            % Remove (or not) outliers
            [XTr, yTr] = filter(XTr, yTr);
            
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
        figure('Name', ['RMSE for ' func2str(splitter) ' + ' func2str(filter)]);
        boxplot(rmse, 'labels', cellfun(@func2str, methods, 'UniformOutput', false));
        title([num2str(k) 'th cluster']);
        %xlabel('methods');
        ylabel('RMSE');
        
    end % clusters
    
end % splitter
