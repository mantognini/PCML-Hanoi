%
% How is the error distributed with svm?

% Load the data
clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
load 'data/data.mat';

%% Binary task

% Split into (Tr, Te)
ratio = 0.7;
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

% Apply svm on {hog, cnn}
tr.y = toBinary(tr.y);
yPredHog = svmF(tr.X.hog, tr.y, val.X.hog, @rbfKernel, 2, 0.00023);
yPredCnn = svmPca(tr.X.cnn, tr.y, val.X.cnn, 150, @rbfKernel, 3.25, 0.00023);

% Errors indices
oIdxHog = (yPredHog < 0);
yPredHog2(oIdxHog) = 0;
yPredHog2(~oIdxHog) = 1;
errIdxHog = (yPredHog2' ~= val.y);

oIdxCnn = (yPredCnn < 0);
yPredCnn2(oIdxCnn) = 0;
yPredCnn2(~oIdxCnn) = 1;
errIdxCnn = (yPredCnn2' ~= val.y);

% Error scores
errScoresHog = yPredHog(errIdxHog);
errScoresCnn = yPredHog(errIdxCnn);

% Plot
figure('Name', 'errors vs confidence');
histogram(errScoresHog, 20);
hold on;
histogram(errScoresCnn, 20);
legend({'hog', 'cnn'});
hold off;

    