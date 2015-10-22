%% Load data
clear all;
load('HaNoi_regression.mat');

% Normalize data
X_trainN = normalize(X_train);
X_testN  = normalize(X_test);


%% Display all plots
nbDim = size(X_train, 2);
plotDim = [4, 4];
plotPerFig = (plotDim(1)*plotDim(2));
nbFig = ceil(size(X_train, 2) / plotPerFig);

assert(nbDim < nbFig * plotPerFig);

for figNo = 0:(nbFig - 1)
    figure('Name', ['raw data, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            plot(X_trainN(:, plotNo), y_train, '.');
            title(['feature #' num2str(plotNo)]);
        end
    end
end


%% Display histogram of response
figure('Name', 'histogram of response');
hist(y_train, 100);
xlabel('y');
ylabel('occurrences');
title('histogram of training data');


%% Display histograms per feature

nbDim = size(X_train, 2);
plotDim = [4, 4];
plotPerFig = (plotDim(1) * plotDim(2));
nbFig = ceil(size(X_train, 2) / plotPerFig);

assert(nbDim < nbFig * plotPerFig);

for figNo = 0:(nbFig - 1)
    figure('Name', ['histograms, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);
    
    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            hist(X_trainN(:, plotNo), 100);
            title(['feature #' num2str(plotNo)]);
        end
    end
end


%% Plot histogram of Discrete features only
groupB     = [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61];
groupBcats = [3,  3,  2,  2,  2,  3,  3,  4,  4,  3,  4,  3];

figure('Name', 'Discrete feature histograms');

for i = 1:length(groupB)
    feature = groupB(i);
    bins = groupBcats(i);
    subplot(3, 4, i);
    hist(X_train(:, feature), bins);
    title([num2str(feature) 'th feature']);
end


%% Summary: display only interesting plots

groupA = [25, 62]; % Features with clusters, multiple input sources
groupB = [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61]; % Discrete features

figure('Name', 'Summary of data exploration');

for featureIdx = 1:length(groupA)
    feature = groupA(featureIdx);
    
    subplotIdx = (featureIdx - 1) * 2 + 1;
    subplot(2, 2, subplotIdx);
    
    hist(X_train(:, feature), 100);
    xlabel('x');
    title(['feature #' num2str(feature)]);
    
    subplot(2, 2, subplotIdx + 1);
    plot(X_train(:, feature), y_train, '.');
    title(['feature #' num2str(feature)]);
end


figure('Name', 'Summary of data exploration (normalized)');

for featureIdx = 1:length(groupA)
    feature = groupA(featureIdx);
    
    subplotIdx = (featureIdx - 1) * 2 + 1;
    subplot(2, 2, subplotIdx);
    
    hist(X_trainN(:, feature), 100);
    xlabel('x');
    title(['feature #' num2str(feature)]);
    
    subplot(2, 2, subplotIdx + 1);
    plot(X_trainN(:, feature), y_train, '.');
    title(['feature #' num2str(feature)]);
end

% Display histogram of response
figure('Name', 'Histogram of response');
hist(y_train, 100);
xlabel('y');
ylabel('occurrences');
title('histogram of training data');

% 3D plot
X_trainN25 = X_trainN(:, 25);
X_trainN62 = X_trainN(:, 62);
figure('Name', 'Feature 25-62');
plot3(X_trainN25, X_trainN62, y_train, '.');
grid on;


%% K-Means of features 25 & 62

K = 3;
X25 = X_train(:, 25);
X62 = X_train(:, 62);
X = [X25 X62];
C = [-0.33 -0.6; -0.26 1.5; 2.43 1.5];
[idx, C] = kmeans(X, K, 'MaxIter', 1000, 'Distance', 'cityblock', 'Start', C);

% 3D Plot of feature 25 & 62 with clustering
figure('Name', [num2str(K) '-Means of feature 25 & 62']);
for k = 1:K
    plot3(X25(idx == k), X62(idx == k), y_train(idx == k), '.', 'MarkerSize', 15);
    hold on;
end
xlabel('25th feature');
ylabel('62th feature');
zlabel('response');
grid on;


%% clusterdata of features 25 & 62

K = 3;
X25 = X_train(:, 25);
X62 = X_train(:, 62);
X = [X25 X62];
idx = clusterdata(X, 'linkage', 'complete', 'distance', 'cityblock', 'maxclust', K);

% 3D Plot of feature 25 & 62 with clustering
figure('Name', [num2str(K) '-clusterdata of feature 25 & 62']);
for k = 1:K
    plot3(X25(idx == k), X62(idx == k), y_train(idx == k), '.', 'MarkerSize', 15);
    hold on;
end
xlabel('25th feature');
ylabel('62th feature');
zlabel('response');
grid on;


%% fitgmdist of features 25 & 62

K = 3;
X25 = X_train(:, 25);
X62 = X_train(:, 62);
X = [X25 X62];
GMModel = fitgmdist(X, K);
idx = cluster(GMModel, X);

% 3D Plot of feature 25 & 62 with clustering
figure('Name', [num2str(K) '-fitgmdist of feature 25 & 62']);
for k = 1:K
    plot3(X25(idx == k), X62(idx == k), y_train(idx == k), '.', 'MarkerSize', 15);
    hold on;
end
xlabel('25th feature');
ylabel('62th feature');
zlabel('response');
grid on;


%% Clustering by hand of features 25 & 62

X25 = X_train(:, 25);
X62 = X_train(:, 62);

lim62 = 15.75;
idx62 = X62 >= lim62;
lim25 = 15.25;
idx25 = X25 < lim25;
idx = idx62 + (idx25 & idx62) + 1; % values are 1, 2 or 3
K = 3;

figure('Name', 'manual split of feature 25 & 62');
for k = 1:K
    plot3(X25(idx == k), X62(idx == k), y_train(idx == k), '.', 'MarkerSize', 15);
    hold on;
end
xlabel('25th feature');
ylabel('62th feature');
zlabel('response');
grid on;
axis square;


%% Clustering using response for validation of manual splitting of features 25 & 62

K = 3;
X25 = X_train(:, 25);
X62 = X_train(:, 62);
X = [X25 X62 y_train];
idx_validation = kmeans(X, K, 'MaxIter', 1000, 'Distance', 'cityblock');
%idx_validation = clusterdata(X, 'linkage', 'average', 'distance', 'cityblock', 'maxclust', K);
%idx_validation = cluster(fitgmdist(X, K), X); % this variante can achieve better results but can be unstable

figure('Name', 'Clustering using response');
for k = 1:K
    plot3(X25(idx_validation == k), X62(idx_validation == k), y_train(idx_validation == k), '.', 'MarkerSize', 15);
    hold on;
end
xlabel('25th feature');
ylabel('62th feature');
zlabel('response');
grid on;
axis square;

% Results: (indexes might need to be swapped before comparing clustering)
diffs = length(find(idx_validation ~= idx));
for i = 1:3
    for j = 1:3
        for k = 1:3
            idx_man = idx;
            idx_man(idx == i) = k;
            idx_man(idx == j) = i;
            idx_man(idx == k) = j;
            diffs_t = length(find(idx_validation ~= idx_man));
            diffs = min(diffs, diffs_t);
        end
    end
end
fprintf(['diffs = ' num2str(diffs) '.\n']); % ~30-50


%% Plot all features with clustered data

nbDim = size(X_train, 2);
plotDim = [4, 4];
plotPerFig = (plotDim(1) * plotDim(2));
nbFig = ceil(size(X_train, 2) / plotPerFig);

assert(nbDim < nbFig * plotPerFig);

for figNo = 0:(nbFig - 1)
    figure('Name', ['clustered raw data, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                plot(X_train(idx == k, plotNo), y_train(idx == k), '.');
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end


%% Reload (splitted/with/without_outliers) data using custom function

clear all;
data = loadRegressionData();

%% Print train & test data in histogramm form using source splitting
K = 3;
nbDim = size(data.dirty.test.X{1}, 2);
plotDim = [4, 4];
plotPerFig = plotDim(1) * plotDim(2);
nbFig = ceil(nbDim / plotPerFig);

assert(nbDim < nbFig * plotPerFig);

for figNo = 0:(nbFig - 1)
    figure('Name', ['clustered train data, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                X = data.dirty.train.X{k};
                y = data.dirty.train.y{k};
                feature = X(:, plotNo);
                plot(feature, y, '.');
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end

for figNo = 0:(nbFig - 1)
    figure('Name', ['histogram of test data, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                X = data.dirty.test.X{k};
                feature = X(:, plotNo);
                histogram(feature);
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end

for figNo = 0:(nbFig - 1)
    figure('Name', ['histogram of train data, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                X = data.dirty.train.X{k};
                feature = X(:, plotNo);
                histogram(feature);
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end

% Display histogram of response
figure('Name', 'Histogram of response');
for k = 1:K
    y = data.dirty.train.y{k};
    histogram(y);
    hold on;
end
xlabel('y');
ylabel('occurrences');
title('histogram of training data');


%% sanity checks about outliers removal
K = 3;

% sanity checks
for k = 1:K
    % Same number of features as dirty, but != #data points
    assert(size(data.dirty.train.X{k}, 2) == size(data.clean.train.X{k}, 2));
    assert(size(data.dirty.train.X{k}, 1) ~= size(data.clean.train.X{k}, 1));
    assert(size(data.dirty.train.Xnorm{k}, 2) == size(data.clean.train.Xnorm{k}, 2));
    assert(size(data.dirty.train.Xnorm{k}, 1) ~= size(data.clean.train.Xnorm{k}, 1));
    assert(size(data.dirty.train.y{k}, 2) == size(data.clean.train.y{k}, 2));
    assert(size(data.dirty.train.y{k}, 1) ~= size(data.clean.train.y{k}, 1));
    
    % Same number of data points between clean training data
    assert(size(data.clean.train.y{k}, 1) == size(data.clean.train.X{k}, 1) && ...
        size(data.clean.train.X{k}, 1) == size(data.clean.train.Xnorm{k}, 1));
end

% cleanup local variables
clear k;
clear K;

%% plot clusters without outliers (manual split)
K = 3;

% ... 2D
figure('Name', '2D clusters without outliers (manual split)');
for k = 1:K
    plot(data.clean.train.X{k}(:, 25), data.clean.train.X{k}(:, 62), ...
        '.', 'MarkerSize', 15);
    hold on;
end

% ... 3D
figure('Name', '3D clusters without outliers (manual split)');
for k = 1:K
    plot3(data.clean.train.X{k}(:, 25), data.clean.train.X{k}(:, 62), ...
        data.clean.train.y{k}, '.', 'MarkerSize', 15);
    hold on;
end

% cleanup local variables
clear k;
clear K;
