%

clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
addpath(genpath('toolboxs/'));
load 'data/data.mat';

ratio = 0.7;

% define method
method = @(train, XValid) rbfSvmPcaCnnManualTree4(train, XValid, -1);


% Split the data
N = size(data.yTrain, 1);
splitIdx = floor(N * ratio);

idx = randperm(N);
idxTrain = idx(1:splitIdx);
idxValid = idx(splitIdx + 1:end);

tr.X.hog = data.hog.train.X(idxTrain, :);
tr.X.cnn = data.cnn.train.X(idxTrain, :);
tr.cnn.mu = data.cnn.mu;
tr.hog.mu = data.hog.mu;
tr.cnn.sigma = data.cnn.sigma;
tr.hog.sigma = data.hog.sigma;
tr.y = data.yTrain(idxTrain); % train y are 4-class

val.X.hog = data.hog.train.X(idxValid, :);
val.X.cnn = data.cnn.train.X(idxValid, :);
val.y = data.yTrain(idxValid);

% apply method
yPred = method(tr, val.X);

%%
% plotConfusion(val.y, int8(yPred));

figure('Name', 'confusion matrix for binary prediction');
mat = confusionmat(val.y, int8(yPred));%rand(2);           %# A 5-by-5 matrix of random values from 0 to 1
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(hot));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
xlabel('Prediction');
ylabel('Truth');
title('Confusion map for binary prediction');

textStrings = num2str(mat(:),'%.f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:4);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

names{1} = 'plane';
names{2} = 'car';
names{3} = 'horse';
names{4} = 'other';

set(gca,'XTick',1:4,...                         %# Change the axes tick marks
        'XTickLabel',names,...  %#   and tick labels
        'YTick',1:4,...
        'YTickLabel',names,...
        'TickLength',[0 0]);

