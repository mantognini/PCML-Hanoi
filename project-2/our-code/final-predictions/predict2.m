%
% Save final predictions for the binary task
clearvars;
addpath(genpath('data/'));
load 'data/data.mat';

% define method
method = @randM2;

% define training data
tr.X.hog = data.hog.train.X;
tr.X.cnn = data.cnn.train.X;
tr.cnn.mu = data.cnn.mu;
tr.hog.mu = data.hog.mu;
tr.cnn.sigma = data.cnn.sigma;
tr.hog.sigma = data.hog.sigma;
tr.y = data.yTrain; % train y are 4-class

% todo replace with testing data
tmpIdx = 1:1:1000;
teX.hog = data.hog.train.X(tmpIdx, :);
teX.cnn = data.cnn.train.X(tmpIdx, :);

% apply method
yPred = method(tr, teX);

% save yTestPred
save 'pred_binary.mat' yPred;

