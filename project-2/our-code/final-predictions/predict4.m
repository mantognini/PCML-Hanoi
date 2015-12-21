%
% Save final predictions for the binary task
clearvars;
addpath(genpath('data/'));
load 'data/data.mat';
load 'data/test.mat';

tr.X.hog = data.hog.train.X;
tr.X.cnn = data.cnn.train.X;
tr.cnn.mu = data.cnn.mu;
tr.hog.mu = data.hog.mu;
tr.cnn.sigma = data.cnn.sigma;
tr.hog.sigma = data.hog.sigma;
tr.y = data.yTrain; % train y are 4-class

te.cnn = normalize(test.X_cnn, tr.cnn.mu, tr.cnn.sigma);
te.hog = normalize(test.X_hog, tr.hog.mu, tr.hog.sigma);

% define method
method = @rbfSvmPcaCnnManualTree4;

% apply method
Ytest = method(tr, te);

% save yTestPred
save 'pred_multiclass.mat' Ytest;

fprintf('done\n');

%%

names{1} = 'plane';
names{2} = 'car';
names{3} = 'horse';
names{4} = 'other';

labels = categorical(Ytest, 1:4, names);

figure;
histogram(labels, 'Normalization', 'probability');



