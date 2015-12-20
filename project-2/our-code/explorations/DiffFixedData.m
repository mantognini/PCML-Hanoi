% Visualise the difference between the original data and the fixed one

clear;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
addpath(genpath('toolboxs/'));
load 'data/data.mat';
load 'data/fixedData.mat';

diffIdx = find(fixedData.yTrain ~= data.yTrain);
fprintf(['# difference = ' num2str(length(diffIdx)) '\n']);

names{1} = 'plane';
names{2} = 'car';
names{3} = 'horse';
names{4} = 'other';
values.plane = 1;
values.car = 2;
values.horse = 3;
values.other = 4;

for cat = 1:4
    catIdx = find(fixedData.yTrain == cat);
    diffIdxCat = intersect(catIdx, diffIdx);
    
    figure('Name', ['Category ' names{cat}]);
    
    subplotIdx = 1;
    suplotSize = ceil(sqrt(length(diffIdxCat)));
    
    for j = 1:length(diffIdxCat)
        i = diffIdxCat(j);

        subplot(suplotSize, suplotSize, subplotIdx);

        img = imread( sprintf('train/imgs/train%05d.jpg', i) );
        imshow(img);
        title([num2str(i) '-th, ' names{fixedData.yTrain(i)} ' vs ' names{data.yTrain(i)}]);

        subplotIdx = subplotIdx + 1;
    end
end

