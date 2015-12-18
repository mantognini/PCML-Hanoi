%
% Benchmark the different learning methods
% reliably on several seeds
% with visualizations
%
% Organized in sections for convenience

clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
load 'data/data.mat';

%% settings
nbRuns = 5;
ratio = 0.7;
category = 1; % for binary mapping


%% Evaluating binary methods
methods2 = {
    @randM2
};
error2 = zeros(nbRuns, length(methods2));

ticId = ticStatus('Evaluating binary methods');
for r = 1:nbRuns
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
        method = methods2{m};

        yPred = method(tr, val.X, category);
        
        % Compute the error
        error2(r, m) = BER(yPred, val.y);
        
        pause(0.1); % so that plot can be displayed
    end
    
    tocStatus(ticId, r / nbRuns);
end

% Plotting BINARY scores
figure('Name', 'BER BINARY');
labels2 = cellfun(@func2str, methods2, 'UniformOutput', false);
boxplot(error2, 'labels', labels2);
title(['category = ' num2str(category)]);


%% Evaluating multi-class methods
methods4 = {
    @randM4,
%     @pcaNnHog4,
%     @pcaNnCnn4,
};
error4 = zeros(nbRuns, length(methods4));

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
        method = methods4{m};

        yPred = method(tr, val.X);
        
        % Compute the errors
        error4(r, m) = BER(yPred, val.y);
        
        pause(0.1); % so that plot can be displayed
    end
end

% Plotting MULTICLASS scores
figure('Name', 'BER MULTICLASS');
labels4 = cellfun(@func2str, methods4, 'UniformOutput', false);
boxplot(error4, 'labels', labels4);
