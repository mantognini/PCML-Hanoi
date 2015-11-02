%% Explore strategies

clear all;
data = loadClassificationData();

% Settings
seeds = 3;         % number of seed to be tested
splitRatio = 0.7;   % training-validation ratio per cluster

strategies = {
    {
        @noClusterSplitter,
        @noFeatureTransformation,
        @noFilter,
        {{
            {@dummyMethod, 'dummy'},
            {@naiveMethod, 'naive'},
            {@LRLSManualMethod, 'logReg+Manual'},
        }}
    },
    {
        @noClusterSplitter,
        @noFeatureTransformation,
        @outliersFilter,
        {{
            {@LRLSManualMethod, 'logReg+Manual-out'},
        }}
    },
    {
        @noClusterSplitter,
        @dummyAndNorm,
        @outliersFilter,
        {{
            {@LRLSMethod, 'logReg-out'},
            {@PLRLSMethod, 'PLRLS-out'},
%             {@PLRLSNewtonMethod, 'PLRLS-out+Newton'},
        }}
    },
    {
        @noClusterSplitter,
        @dummyAndNormAndGaussian,
        @outliersFilter,
        {{
            {@LRLSMethod, 'logReg-out+^1/2'},
%             {@PLRLSMethod, 'PLRLS-out+1/2'},
%             {@PLRLSNewtonMethod, 'PLRLS-out+1/2+New'},
        }}
    },
};

% Compute & plot misclassification for strategies
finalPlotError = [];
finalPlotLabels = {};
for splitterNo = 1:numel(strategies)
    splitter = strategies{splitterNo}{1};
    featureTransformation = strategies{splitterNo}{2};
    filter = strategies{splitterNo}{3};
    
    [K, clusters] = splitter(data);
    
    % Test all methods on each clusters
    for k = 1:K
        methods = strategies{splitterNo}{4}{k};
        cluster = featureTransformation(clusters{k});
        
%         % plot feature repartition
%         figure;
%         boxplot(cluster.train.X);
        
        error = zeros(seeds, numel(methods));
        
        N = size(cluster.train.X, 1);
        labels = {};
    
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
                method = methods{methodNo}{1};
                labels{methodNo} = methods{methodNo}{2};
                
                % Collect predictions
                yValidPred = method(XTr, yTr, XValid);
                
                % Compute error
                error(seed, methodNo) = zeroOneLoss(yValid, yValidPred);
            end % methods

        end % seeds
        
        % Plots data
        finalPlotError = [finalPlotError error];
        finalPlotLabels = [finalPlotLabels labels];
        
    end % clusters
    
    
end % splitter

% Plots
figure('Name', 'recap');
boxplot(finalPlotError, 'labels', finalPlotLabels);
ylabel('0-1 Loss');

