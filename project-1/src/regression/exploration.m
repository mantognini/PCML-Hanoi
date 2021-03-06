%% Load data
clear all;
load('HaNoi_regression.mat');

% Normalize data
X_trainN = normalize(X_train);
X_testN  = normalize(X_test);


%% Clustering using response on part of the data to determine cluster of rest of the data

X25 = X_train(:, 25);
X62 = X_train(:, 62);
X = [X25 X62];
clear X25 X62;

splitRatio = 0.7;
% seed = 2;
% setSeed(seed);

% Split data into training and validation sets
N = size(X, 1);
idx = randperm(N);
NTr = floor(splitRatio * N);
idxTr = idx(1:NTr);
idxVa = idx(NTr+1:end);
XTr = X(idxTr, :);
yTr = y_train(idxTr);
XVa = X(idxVa, :);
yVa = y_train(idxVa);
clear idx NTr N;

% Clusterize data
K = 3;
% C = [ 12, 12, 1800 ; 12, 18, 5000 ; 17, 18, 8000 ];
% idxTr = kmeans([XTr yTr], K, 'MaxIter', 1000, 'Start', C);
% idxTr = kmeans([XTr yTr], K, 'MaxIter', 1000, 'Distance', 'cityblock');
idxTr = cluster(fitgmdist([XTr yTr], K), [XTr yTr]); % this variante can achieve better results but can be unstable


% Print result
figure('Name', 'Clustering using response');
subplot(1, 2, 1);
mus = zeros(K, 2);
sigmas = zeros(K, 2);
for k = 1:K
    plot3(XTr(idxTr == k, 1), ... % X25, training, cluster k
          XTr(idxTr == k, 2), ... % X62, training, cluster k
          yTr(idxTr == k),    ... % y,   training, cluster k
          '.', 'MarkerSize', 30);
    hold on;
    
    mu25 = mean(XTr(idxTr == k, 1));
    mu62 = mean(XTr(idxTr == k, 2));
    
    std25 = std(XTr(idxTr == k, 1));
    std62 = std(XTr(idxTr == k, 2));
    
    mus(k, :) = [ mu25 , mu62 ];
    sigmas(k, :) = [ std25 , std62 ];
    
%     fprintf(['cluster k = ' num2str(k) ': X25 ->\t ? = ' num2str(mu25) ...
%              '\tstd = ' num2str(std25) '\n']);
%     fprintf(['cluster k = ' num2str(k) ': X62 ->\t ? = ' num2str(mu62) ...
%              '\tstd = ' num2str(std62) '\n']);
         
    clear mu25 mu62 std25 std62;
end
xlabel('25th feature');
xlim([10 20]);
ylabel('62th feature');
ylim([0 30]);
zlabel('response');
title('TRAINING');
grid on;
% axis square;

% Compute probabilites of being in a cluster
pVa = zeros(size(XVa, 1), K);
for k = 1:K
    pVa(:, k) = mvnpdf(XVa, mus(k, :), sigmas(k, :));
end
clear mus sigmas;

[~, idxVa] = max(pVa, [], 2);
clear pVa;

% Print result
subplot(1, 2, 2);
for k = 1:K
    plot3(XVa(idxVa == k, 1), ... % X25, validation, cluster k
          XVa(idxVa == k, 2), ... % X62, validation, cluster k
          yVa(idxVa == k),    ... % y,   validation, cluster k
          '.', 'MarkerSize', 30);
    hold on;
end
xlabel('25th feature');
xlim([10 20]);
ylabel('62th feature');
ylim([0 30]);
zlabel('response');
title('VALIDATION');
grid on;
% axis square;


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
%idx_validation = kmeans(X, K, 'MaxIter', 1000, 'Distance', 'cityblock');
%idx_validation = clusterdata(X, 'linkage', 'average', 'distance', 'cityblock', 'maxclust', K);
idx_validation = cluster(fitgmdist(X, K), X); % this variante can achieve better results but can be unstable

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

%% Print train & test data in histogram form using source splitting without outliers
K = 3;
nbDim = size(data.clean.test.X{1}, 2);
plotDim = [4, 4];
plotPerFig = plotDim(1) * plotDim(2);
nbFig = ceil(nbDim / plotPerFig);

for figNo = 0:(nbFig - 1)
    figure('Name', ['clustered train data without outliers, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                X = data.clean.train.X{k};
                y = data.clean.train.y{k};
                feature = X(:, plotNo);
                plot(feature, y, '.');
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end

for figNo = 0:(nbFig - 1)
    figure('Name', ['histogram of test data without outliers, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                X = data.clean.test.X{k};
                feature = X(:, plotNo);
                histogram(feature);
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end

for figNo = 0:(nbFig - 1)
    figure('Name', ['histogram of train data without outliers, ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);

            for k = 1:K
                X = data.clean.train.X{k};
                feature = X(:, plotNo);
                histogram(feature);
                hold on;
            end
            title(['feature #' num2str(plotNo)]);
        end
    end
end

% Display histogram of response
figure('Name', 'Histogram of response without outliers,');
for k = 1:K
    y = data.clean.train.y{k};
    histogram(y);
    hold on;
end
xlabel('y');
ylabel('occurrences');
title('histogram of training data without outliers');

% cleanup local variables
clear k K nbDim plotDim plotPerFig nbFig;


%% Plot each feature/response per cluster
clear all;

% Load dataset
allData = loadRegressionData();
data = allData.original;
[K, clusters] = manualClusterSplitter(data);

side = 3;

for k = 1:K
    cluster = clusters{k};
    D = size(cluster.train.X, 2);
    
    for f = 1:D
        if (mod(f - 1, side * side) == 0)
            figure('Name', ['Cluster ' num2str(k)]);
        end
        
        subplot(side, side, mod(f - 1, side * side) + 1);
        
        x = cluster.train.X(:, f);
        y = cluster.train.y;
        
        plot(x, y, '.');
        xlabel([num2str(f) 'th feature']);
        ylabel('response');
    end
end

% Non flat features/response plot:
% cluster 1 => 4, 16, 53
% cluster 2 => 14, 18, 20, 43?, 59
% cluster 3 => 5, 11, 33, 46

