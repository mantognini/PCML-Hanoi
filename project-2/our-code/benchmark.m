% Benchmark the different learning methods
% reliably on several seeds
% with visualizations
%
% Organized in sections for convenience

clearvars;
addpath(genpath('data/train/'));
load 'data/train/train.mat';

%% 
methods = {@randomMethod};

% Split the data
N = size(train.y, 1);
ratio = 0.7;
splitIdx = floor(N * ratio);

idx = randperm(N);
idxTrain = idx(1:splitIdx);
idxValid = idx(splitIdx + 1:end);

data.train.X.hog = train.X_hog(idxTrain, :);
data.train.X.cnn = train.X_cnn(idxTrain, :);
data.train.y = train.y(idxTrain);

data.valid.X.hog = train.X_hog(idxValid, :);
data.valid.X.cnn = train.X_cnn(idxValid, :);
data.valid.y = train.y(idxValid);

for m = 1:length(methods)
    method = methods{m};
    
    yPred = method(data.train, data.valid.X);
    error = BER(yPred, data.valid.y);
end

