%
% Benchmark the different learning methods
% reliably on several seeds
% with visualizations
%
% Organized in sections for convenience

clearvars;
addpath(genpath('data/'));
load 'data/data.mat';

% settings
nbRuns = 5;
ratio = 0.7;

%% Evaluating binary methods
% med   25-75   Method
% -------------------
% 0.5           @randM2
% 0.30  0.04    @linSvmHogF2 C = 1
% 0.23  0.05    @linSvmHogCV2 C* = 0.00023
methods2 = {
%     @linSvmHogCV2,
    @randM2
};
error2 = zeros(nbRuns, length(methods2));

for r = 1:nbRuns
    fprintf(['run n?' num2str(r) '/' num2str(nbRuns) '\n']);
    
    % Split the data
    N = size(data.yTrain, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    tr.X.hog = data.hog.train.X(idxTrain, :);
    tr.X.cnn = data.cnn.train.X(idxTrain, :);
    tr.y = data.yTrain(idxTrain); % train y are 4-class

    val.X.hog = data.hog.train.X(idxValid, :);
    val.X.cnn = data.cnn.train.X(idxValid, :);
    val.y = toBinary(data.yTrain(idxValid)); % valid y are binary

    % Run the methods
    for m = 1:length(methods2)
        method = methods2{m};

        yPred = method(tr, val.X);
        
        % Compute the error
        error2(r, m) = BER(yPred, val.y);
    end
end

%% Evaluating multi-class methods
% med   25-75   Method
% -------------------
% 0.75          @randM4
methods4 = {
    @randM4,
    @pcaNnHog4,
    @pcaNnCnn4,
};
error4 = zeros(nbRuns, length(methods4));

for r = 1:nbRuns
    % Split the data
    N = size(data.yTrain, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTrain = idx(1:splitIdx);
    idxValid = idx(splitIdx + 1:end);

    tr.X.hog = data.hog.train.X(idxTrain, :);
    tr.X.cnn = data.cnn.train.X(idxTrain, :);
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
