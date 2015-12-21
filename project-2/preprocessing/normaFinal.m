%
% Normalize the data for the final predictions
clearvars;
addpath(genpath('data/'));
load 'data/test.mat';
load 'data/train/train.mat';

% Normalize the data
[X1, ~, ~] = zscore([test.X_cnn; train.X_cnn]);
[X2, ~, ~] = zscore([test.X_hog; train.X_hog]);

test.X_cnn = X1(size(test.X_cnn, 1), :);
train.X_cnn = X1(size(train.X_cnn, 1), :);

test.X_hog = X2(size(test.X_hog, 1), :);
train.X_hog = X2(size(train.X_hog, 1), :);

% Free space
clear X1,
clear X2;

% training data
data.tr.X.hog = train.X_hog;
data.tr.X.cnn = train.X_cnn;
data.tr.y = train.y;

% testing data
data.te.X.hog = test.X_hog;
data.te.X.cnn = test.X_cnn;

save 'finalData.mat' data;