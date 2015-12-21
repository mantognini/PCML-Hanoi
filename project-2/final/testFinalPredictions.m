%
% Test
clearvars;
addpath(genpath('data/'));
load 'data/data.mat';
load 'data/mock-solution.mat';
load 'data/mock-test.mat';

%%

tr.X.hog = data.hog.train.X;
tr.X.cnn = data.cnn.train.X;
tr.cnn.mu = data.cnn.mu;
tr.hog.mu = data.hog.mu;
tr.cnn.sigma = data.cnn.sigma;
tr.hog.sigma = data.hog.sigma;
tr.y = data.yTrain; % train y are 4-class

te.cnn = normalize(test.X_cnn, tr.cnn.mu, tr.cnn.sigma);
te.hog = normalize(test.X_hog, tr.hog.mu, tr.hog.sigma);


%%

% define method
% C* = 3.25, M* = 150, gamma = 0.00023
method = @(tr, XValid) rbfSvmPcaCnnF2(tr, XValid, 4, 150, 3.25, 0.00023);

% apply method
Ytest = method(tr, te);

BER(toBinary(y, 4), Ytest)

%%
% define method
method = @rbfSvmPcaCnnManualTree4;

% apply method
Ytest = method(tr, te);
BER(y, Ytest)
