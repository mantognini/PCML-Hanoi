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
        @noFeatureTransformation,
        @noFilter,
        %@outliersFilter,
        % method for the unique cluster
        {{
            @constantMethod,
            %@medianMethod,
            @meanMethod,
            @GDLSMethod,
%           @ridgeLinear5Fold,
            @ridgeLinear10Fold,
%           @ridgeLinear20Fold,
            @(XTr, yTr, XValid) ridgeEmplifiedKFold12(10, XTr, yTr, XValid),
%           @(XTr, yTr, XValid) ridgeEmplifiedKFold051(10, XTr, yTr, XValid),
%           @(XTr, yTr, XValid) ridgeEmplifiedKFoldSin1(10, XTr, yTr, XValid),
%           @(XTr, yTr, XValid) ridgeEmplifiedKFold13(10, XTr, yTr, XValid),
%           @(XTr, yTr, XValid) ridgeEmplifiedKFold23(10, XTr, yTr, XValid),
            @(XTr, yTr, XValid) ridgeEmplifiedKFold123(20, XTr, yTr, XValid),
        }}
    },

%     {
%         @manualClusterSplitter,
%         @noFeatureTransformation,
%         @outliersFilter,
%         % method for the 1st cluster
%         {
%             {
%                 %@constantMethod,
%                 %@medianMethod,
%                 @meanMethod,
%                 @GDLSMethod,
% %                 @ridgeLinear5Fold,
%                 @ridgeLinear10Fold,
% %                 @ridgeLinear20Fold,
%                 @(XTr, yTr, XValid) ridgeEmplifiedKFold12(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold051(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFoldSin1(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold13(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold23(10, XTr, yTr, XValid),
%                 @(XTr, yTr, XValid) ridgeEmplifiedKFold123(20, XTr, yTr, XValid),
%             }
%             % method for the 2nd cluster
%             {
%                 %@constantMethod,
%                 %@medianMethod,
%                 @meanMethod,
%                 @GDLSMethod,
% %                 @ridgeLinear5Fold,
%                 @ridgeLinear10Fold,
% %                 @ridgeLinear20Fold,
%                 @(XTr, yTr, XValid) ridgeEmplifiedKFold12(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold051(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFoldSin1(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold13(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold23(10, XTr, yTr, XValid),
%                 @(XTr, yTr, XValid) ridgeEmplifiedKFold123(20, XTr, yTr, XValid),
%             }
%             % method for the 3rd cluster
%             {
%                 %@constantMethod,
%                 %@medianMethod,
%                 @meanMethod,
%                 @GDLSMethod,
% %                 @ridgeLinear5Fold,
%                 @ridgeLinear10Fold,
% %                 @ridgeLinear20Fold,
%                 @(XTr, yTr, XValid) ridgeEmplifiedKFold12(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold051(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFoldSin1(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold13(10, XTr, yTr, XValid),
% %                 @(XTr, yTr, XValid) ridgeEmplifiedKFold23(10, XTr, yTr, XValid),
%                 @(XTr, yTr, XValid) ridgeEmplifiedKFold123(20, XTr, yTr, XValid),
%             }
%        }
%     },
};

% Compute & plot RMSE for each data splitting & model strategies

fprintf('I am starting\n');

% For each splitting strategy
for splitterNo = 1:numel(strategies)
    splitter = strategies{splitterNo}{1};
    featureTransformation = strategies{splitterNo}{2};
    filter   = strategies{splitterNo}{3};
    
    [K, clusters] = splitter(data);
    
    % Test all methods on each clusters
    for k = 1:K
        fprintf(['processing cluster ' num2str(k) ' of ' num2str(K) '\n']);
        methods  = strategies{splitterNo}{4}{k};
        
        if (numel(methods) == 0)
            fprintf(['Skipping clurser ' num2str(k) ' because no method\n']);
            continue;
        end
        
        cluster = featureTransformation(clusters{k});
        
        rmse = zeros(seeds, numel(methods));
        
        N = size(cluster.train.X, 1);
    
        % Test each methods several times with different training and 
        % validation split of the data
        for seed = 1:seeds
            fprintf(['\tprocessing seed ' num2str(seed) ' of ' num2str(seeds) '\n']);
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
                fprintf(['\t\tprocessing method ' num2str(methodNo) ' of ' num2str(numel(methods)) '\n']);
                method = methods{methodNo};
                
                % Collect predictions
                yValidPred = method(XTr, yTr, XValid);
                
                % Compute error
                rmse(seed, methodNo) = computeRmse(yValidPred - yValid);
            end % methods

        end % seeds
        
        % Plot RMSE for this cluster
        figure('Name', ['RMSE for ' func2str(splitter) ' + ' ...
            func2str(featureTransformation) ' + ' func2str(filter)]);
        boxplot(rmse);%, 'labels', cellfun(@func2str, methods, 'UniformOutput', false));
        %set(gca, 'FontSize', 10, 'XTickLabelRotation', 90);
        title([num2str(k) 'th cluster']);
        %xlabel('methods');
        ylabel('RMSE');
        
    end % clusters
    
end % splitter

fprintf('I am finished\n');

