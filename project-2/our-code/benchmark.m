% Benchmark the different learning methods
% reliably on several seeds
% with visualizations
%
% Organized in sections for convenience

clearvars;
addpath(genpath('data/train/'));
load 'data/train/train.mat';

% avoid future bugs
train.X_cnn = single(train.X_cnn);
train.X_hog = single(train.X_hog);
train.y = single(train.y);

% settings
nbRuns = 2;
ratio = 0.7;

%% Evaluating binary methods
% med   25-75   Method
% -------------------
% 0.5           @randM2
% 0.30  0.04    @linSvmHog2
methods2 = {
    @randM2,
    @linSvmHog2
    };
error2 = zeros(nbRuns, length(methods2));

for r = 1:nbRuns
    % Split the data
    N = size(train.y, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    data.train.X.hog = train.X_hog(idxTrain, :);
    data.train.X.cnn = train.X_cnn(idxTrain, :);
    data.train.y = train.y(idxTrain); % train y are 4-class

    data.valid.X.hog = train.X_hog(idxValid, :);
    data.valid.X.cnn = train.X_cnn(idxValid, :);
    data.valid.y = toBinary(train.y(idxValid)); % valid y are binary

    for m = 1:length(methods2)
        method = methods2{m};

        yPred = method(data.train, data.valid.X);
        error2(r, m) = BER(yPred, data.valid.y);
    end
end

%% Evaluating multi-class methods
% med   25-75   Method
% -------------------
% 0.75          @randM4
methods4 = {
    @randM4
    };
error4 = zeros(nbRuns, length(methods4));

for r = 1:nbRuns
    % Split the data
    N = size(train.y, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    data.train.X.hog = train.X_hog(idxTrain, :);
    data.train.X.cnn = train.X_cnn(idxTrain, :);
    data.train.y = train.y(idxTrain); % train y are 4-class

    data.valid.X.hog = train.X_hog(idxValid, :);
    data.valid.X.cnn = train.X_cnn(idxValid, :);
    data.valid.y = train.y(idxValid); % valid y are 4-class

    for m = 1:length(methods4)
        method = methods4{m};

        yPred = method(data.train, data.valid.X);
        error4(r, m) = BER(yPred, data.valid.y);
    end
end

%% Plotting scores
figure('Name', 'BER');

subplot(1, 2, 1);
labels2 = cellfun(@func2str, methods2, 'UniformOutput', false);
boxplot(error2, 'labels', labels2);

subplot(1, 2, 2);
labels4 = cellfun(@func2str, methods4, 'UniformOutput', false);
boxplot(error4, 'labels', labels4);
