%
% Save final predictions for the binary task
clearvars;
addpath(genpath('data/'));
load 'data/finalData.mat';

% define method
method = @rbfSvmPcaCnnManualTree4;

% apply method
Ytest = method(data.tr, data.te);

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



