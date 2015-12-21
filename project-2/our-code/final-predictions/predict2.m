%
% Save final predictions for the binary task
clearvars;
addpath(genpath('data/'));
load 'data/finalData.mat';

% define method
% C* = 3.25, M* = 150, gamma = 0.00023
method = @(tr, XValid) rbfSvmPcaCnnF2(tr, XValid, 4, 150, 3.25, 0.00023);

% apply method
Ytest = method(data.tr, data.te);

% save yTestPred
save 'pred_binary.mat' Ytest;

fprintf('done\n');
