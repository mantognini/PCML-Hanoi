%
% Benchmark the different learning methods
% reliably on several seeds
% with visualizations
%
% Organized in sections for convenience

clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
addpath(genpath('toolboxs/'));
load 'data/data.mat';

%% settings
nbRuns = 20;
ratio = 0.7;
category = 4; % for binary mapping


%% Evaluating binary methods
methods2 = {
    { 'log Reg HOG', @logRegHog2 }, % 1
    { 'NN HOG', @nnHog2 }, % 3
    { 'lin SVM HOG', @(x, y, c) linSvmHogF2(x, y, c, 0.00023) }, % 4
    { 'rbf SVM HOG', @(x, y, c) rbfSvmHogF2(x, y, c, 2, 0.00023) }, % 5
    { 'RF CNN', @cnnForestF2 }, %2
    { 'NN CNN', @nnCnn2 }, % 6
    { 'log reg CNN', @logRegCnn2 }, % 7
    { 'rbf SVM MC', @svmHogCnnMC2 }, % 8
    { 'lin SVM CNN', @(x, y, c) linSvmPcaCnnF2(x, y, c, 0.00023, 1300) }, % 9
    { 'rbf SVM CNN', @(x, y, c) rbfSvmPcaCnnF2(x, y, c, 150, 3.25, 0.00023) } % 10
};

labels2 = cellfun(@(m) m{1}, methods2, 'UniformOutput', 0);

%%
% error2 = zeros(nbRuns, length(methods2));
for r = 1:nbRuns
    fprintf(['run ' num2str(r) '/' num2str(nbRuns) '\n']);
    
    % Split the data
    N = size(data.yTrain, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    tr.X.hog = data.hog.train.X(idxTrain, :);
    tr.X.cnn = data.cnn.train.X(idxTrain, :);
    tr.cnn.mu = data.cnn.mu;
    tr.hog.mu = data.hog.mu;
    tr.cnn.sigma = data.cnn.sigma;
    tr.hog.sigma = data.hog.sigma;
    tr.y = data.yTrain(idxTrain); % train y are 4-class

    val.X.hog = data.hog.train.X(idxValid, :);
    val.X.cnn = data.cnn.train.X(idxValid, :);
    val.y = toBinary(data.yTrain(idxValid), category); % valid y are binary

    % Run the methods
    for m = 1:length(methods2)
        method = methods2{m}{2};
        label = methods2{m}{1};

        yPred = method(tr, val.X, category);
        
        % Compute the error
        err = BER(yPred, val.y);
        error2(r, m) = err;
        
        fprintf([label ' gave BER = ' num2str(err) '\n']);
        pause(0.1); % so that plot can be displayed
    end
end

%%
% Plotting BINARY scores
figure('Name', 'BER BINARY');
boxplot(error2, 'labels', labels2');%, 'labelorientation', 'inline');
title(['category = ' num2str(category)]);


%% Evaluating multi-class methods
methods4 = {
%     { 'NN HOG', @pcaNnHog4 }, % 0.30
    { 'NN CNN', @pcaNnCnn4 }, % 0.095
    { 'SVM Matlab', @svmPcaCnnMatlab4 } % 0.085
    { 'SVM + Manual Tree', @(train, XValid) rbfSvmPcaCnnManualTree4(train, XValid, -1) }, % 0.078
    { 'SVM + Manual Tree 2', @(train, XValid) rbfSvmPcaCnnManualTree4(train, XValid, -0.7) }, % 0.072
};

labels4 = cellfun(@(m) m{1}, methods4, 'UniformOutput', 0);

%%
% error4 = zeros(nbRuns, length(methods4));

for r = 1:nbRuns
    fprintf(['run ' num2str(r) '/' num2str(nbRuns) '\n']);

    % Split the data
    N = size(data.yTrain, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    tr.X.hog = data.hog.train.X(idxTrain, :);
    tr.X.cnn = data.cnn.train.X(idxTrain, :);
    tr.cnn.mu = data.cnn.mu;
    tr.hog.mu = data.hog.mu;
    tr.cnn.sigma = data.cnn.sigma;
    tr.hog.sigma = data.hog.sigma;
    tr.y = data.yTrain(idxTrain); % train y are 4-class

    val.X.hog = data.hog.train.X(idxValid, :);
    val.X.cnn = data.cnn.train.X(idxValid, :);
    val.y = data.yTrain(idxValid); % valid y are 4-class

    % Run the methods
    for m = 1:length(methods4)
        method = methods4{m}{2};
        label = methods4{m}{1};

        yPred = method(tr, val.X);
        
        % Compute the errors
        err = BER(yPred, val.y);
        error4(r, m) = err;
        
        fprintf([label ' gave BER = ' num2str(err) '\n']);
        pause(0.1); % so that plot can be displayed
    end
end


%% Plotting MULTICLASS scores
figure('Name', 'BER MULTICLASS');
boxplot(error4, 'labels', labels4');
ylabel('BER');
title('Method performance for multiclass models');

mean(error4)

