%% Load data
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

% Display histogram of response
figure('Name', 'histogram of response');
hist(y_train, 100);
xlabel('y');
ylabel('occurrences');
title('histogram of training data');


% Display histograms per feature
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

%% 3-mean: identification of the three input sources

X_trainN25 = X_trainN(:, 25);
X_trainN62 = X_trainN(:, 62);
X = [X_trainN25 X_trainN62];
C = [-0.33 -0.6; -0.26 1.5; 2.43 1.5];
[idx, C] = kmeans(X, 3, 'MaxIter',1000, 'Start', C);

X_25k1 = X_trainN25(idx == 1);
X_25k2 = X_trainN25(idx == 2);
X_25k3 = X_trainN25(idx == 3);
X_62k1 = X_trainN62(idx == 1);
X_62k2 = X_trainN62(idx == 2);
X_62k3 = X_trainN62(idx == 3);
y_k1 = y_train(idx == 1);
y_k2 = y_train(idx == 2);
y_k3 = y_train(idx == 3);

figure;
subplot(1, 2, 1);
plot(X_25k1, y_k1, 'r.', X_25k2, y_k2, 'g.', X_25k3, y_k3, 'b.', 'MarkerSize', 15);
grid on;
subplot(1, 2, 2);
plot(X_62k1, y_k1, 'r.', X_62k2, y_k2, 'g.', X_62k3, y_k3, 'b.', 'MarkerSize', 15);
grid on;

figure;
plot3(X_25k1, X_62k1, y_k1, 'r.', 'MarkerSize', 15);
hold on;
plot3(X_25k2, X_62k2, y_k2, 'g.', 'MarkerSize', 15);
plot3(X_25k3, X_62k3, y_k3, 'b.', 'MarkerSize', 15);
hold off;
grid on;

