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

% settings
nbRuns = 5;
ratio = 0.7;

%% Evaluating binary methods
% med   25-75   Method
% -------------------
% 0.5           @randM2
% rbfSvmHogF2
% 0.39  0.015   @rbfSvmHogF2 C = 0.00023, gamma = 1
% 0.30  0.04    @linSvmHogF2 C = 1
% 0.26  0.02    @nnHog2
% 0.25  0.02    @pcaNnHog2
% 0.23  0.05    @linSvmHogCV2 C* = 0.00023
% 0.18  0.01    @rbfSvmHogCV2 C* = 2, gamma* = 0.00023
% 0.112 0.11    @nnCnn2
% 0.104 0.01    @pcaNnCnn2
% 0.095 0.004   @svmHogCnnKNN2, split = 0.9, k = 5
% 0.089 0.008   @svmHogCnnMC2  
% 0.087 0.009   @linSvmPcaCnnCV2 C* = 0.00023, M* = 1300
% 0.081 0.008   @rbfSvmPcaCnnF2 C* = 3.25, M* = 150, gamma = 0.00023
% 0.079 0.004   @svmHogCnnCustom2
methods2 = {
%     @randM2,
%     @linSvmHogCV2,
%     @rbfSvmHogCV2,
%     @linSvmPcaHogCV2,
%     @linSvmPcaCnnCV2,
%     @rbfSvmPcaCnnCV2
%     @pcaNnHog2,
%     @pcaNnCnn2,
%     @nnHog2,
%     @nnCnn2,
%     @svmHogCnnMC2
    @svmHogCnnKNN2
%     @svmHogCnnCustom2
};
error2 = zeros(nbRuns, length(methods2));

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
    val.y = toBinary(data.yTrain(idxValid)); % valid y are binary

    % Run the methods
    for m = 1:length(methods2)
        method = methods2{m};

        yPred = method(tr, val.X);
        
        % Compute the error
        error2(r, m) = BER(yPred, val.y);
        
        pause(0.1); % so that plot can be displayed
    end
end

%% Evaluating multi-class methods
% med   25-75   Method
% -------------------
% 0.75          @randM4
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

%% Plotting scores
figure('Name', 'BER');

% subplot(1, 2, 1);
labels2 = cellfun(@func2str, methods2, 'UniformOutput', false);
boxplot(error2, 'labels', labels2);
ylim([0 0.15]);

% subplot(1, 2, 2);
% labels4 = cellfun(@func2str, methods4, 'UniformOutput', false);
% boxplot(error4, 'labels', labels4);
