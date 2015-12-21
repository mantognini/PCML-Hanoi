%
% Normalize training data

clearvars;
addpath(genpath('data/train/'));
load 'data/train/train.mat';

% normalize
[data.cnn.train.X, muCnn, sigmaCnn] = zscore(train.X_cnn);
[data.hog.train.X, muHog, sigmaHog] = zscore(train.X_hog);
data.yTrain = train.y;

% save distribution parameters
data.cnn.mu = muCnn;
data.hog.mu = muHog;

data.cnn.sigma = sigmaCnn;
data.hog.sigma = sigmaHog;

% save to file
save 'data/data.mat' data;

