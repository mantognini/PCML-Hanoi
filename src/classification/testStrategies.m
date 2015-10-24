%% Explore strategies

clear all;
data = loadClassificationData();

% Settings
seeds = 20;         % number of seed to be tested
splitRatio = 0.7;   % training-validation ratio per cluster

strategies = {
    {
        @noClusterSplitter,
        @noFeatureTransformation,
        @noFilter,
        % method for the unique cluster
        {{
            @dummyClassificationMethod,
            @naiveClassificationMethod,
            @logisticClassificationMethod,
            @logisticClassificationWithManualSplittingMethod,
        }}
    },
};

% Compute & plot misclassification for strategies

for splitterNo = 1:numel(strategies)
    splitter = strategies{splitterNo}{1};
    featureTransformation = strategies{splitterNo}{2};
    filter = strategies{splitterNo}{3};
    
    [K, clusters] = splitter(data);
    
    % Test all methods on each clusters
    for k = 1:K
        methods = strategies{splitterNo}{4}{k};
        cluster = featureTransformation(clusters{k});
        
        error = zeros(seeds, numel(methods));
        
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
                mismatch = length(find(yValidPred ~= yValid));
                error(seed, methodNo) = mismatch / length(yValidPred);
            end % methods

        end % seeds
        
        % Plot RMSE for this cluster
        figure('Name', ['Misclassification for ' func2str(splitter) ' + ' ...
            func2str(featureTransformation) ' + ' func2str(filter)]);
        boxplot(error, 'labels', cellfun(@func2str, methods, 'UniformOutput', false));
        title([num2str(k) 'th cluster']);
        %xlabel('methods');
        ylabel('misclassification');
        
    end % clusters
    
end % splitter

