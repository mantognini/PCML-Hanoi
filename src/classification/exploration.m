%% Explore classification data set

%% Load data & some general facts

clear all;
load('HaNoi_classification');

% X_test    is 1500x40
% X_train   is 1500x40
% y_train   is 1500x1  which values are either -1 or 1

N = length(y_train);
D = size(X_train, 2);

% Feature-wise, we have 5 discrete input variables:
discreteFeatures = [ 16, 25, 26, 32, 38 ];
% # of categories:    4   2   4   2   2
% They all have a relatively uniform distribution among categories,
% especially feature 38 which has 50%-50% distribution for both test
% and train data.

% Continuous input
continuousFeatures = setdiff([1:D], discreteFeatures);

% Features 1, 5, 11 and 36 (less cleanly) have a gaussian distribution
% plus a spike outside the gaussian curve's main domain.
keyContinuousFeatures = [1, 5, 11, 36];
% Bivariate distributions show that there might indeed be something about
% this spike. 


%% Display histogram of response
figure('Name', 'Histogram of response');
y_categorical = categorical(y_train); % convert {-1, 1} to categories "-1" and "1"
h = histogram(y_categorical);
h.Normalization = 'probability';
xlabel('output categories');
ylabel('probability');
ylim([0 1]);


%% Display only discrete features
figure('Name', 'Distributions of discrete features (train + test)');
for i = 1:length(discreteFeatures)
    subplot(2, 3, i);
    f = discreteFeatures(i);
    
    plotFeature(true, f, X_train, X_test);
end


%% Display only discrete features with respect to the response
figure('Name', 'Distributions of discrete features with the response');
for i = 1:length(discreteFeatures)
    subplot(2, 3, i);
    f = discreteFeatures(i);
    
    plotFeatureResponse(f, X_train, y_train);
end

% For reference:
subplot(2, 3, 6);
y_categorical = categorical(y_train); % convert {-1, 1} to categories "-1" and "1"
h = histogram(y_categorical);
h.Normalization = 'probability';
xlabel('output categories');
ylabel('probability');
%ylim([0 1]);


%% Display only continuous features
figures = length(continuousFeatures);
figNo = 0;
for i = 1:figures
    if (mod(i - 1, 9) == 0)
        figNo = figNo + 1;
        figure('Name', ['Histograms of continuous features (train + test) ' num2str(figNo) ' / 4']);
    end
    subplot(3, 3, mod(i - 1, 9) + 1);
    f = continuousFeatures(i);
    
    plotFeature(false, f, X_train, X_test);
end


%% Display bivariate histograms of key features with response
figure('Name', 'Histograms of key features with response');
for i = 1:length(keyContinuousFeatures);
    f = keyContinuousFeatures(i);
    
    subplot(2, 2, i);
    
    plotFeatureResponse(f, X_train, y_train);
end


%% Explore strategies

clear all;
data = loadClassificationData();

% Settings
seeds = 10;         % number of seed to be tested
splitRatio = 0.7;   % training-validation ratio per cluster

strategies = {
    {
        @noClusterSplitter,
        @noFeatureTransformation,
        @noFilter,
        % method for the unique cluster
        {{
            @dummyClassificationMethod,
        }}
    },
};

fprintf('TODO we want classification here, not regression\n');

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
                error(seed, methodNo) = length(find(yValidPred ~= yValid));
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
