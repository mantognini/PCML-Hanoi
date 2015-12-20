% Script used to generate a few figures
%



%% HOG ILLUSTRATION

addpath(genpath('data/train/'));
load 'data/train/train.mat';

figure;

i = 20;
subplot(2, 4, 1);
img = imread( sprintf('train/imgs/train%05d.jpg', i) );
imshow(img);
subplot(2, 4, 2);
f = train.X_hog(i, :);
f = reshape(f, 13, 13, 32); % Those dimensions correspond to hog( single(img)/255, 17, 8);
im( hogDraw(f) ); colormap gray;
axis off; colorbar off;


i = 16;
subplot(2, 4, 3);
img = imread( sprintf('train/imgs/train%05d.jpg', i) );
imshow(img);
subplot(2, 4, 4);
f = train.X_hog(i, :);
f = reshape(f, 13, 13, 32); % Those dimensions correspond to hog( single(img)/255, 17, 8);
im( hogDraw(f) ); colormap gray;
axis off; colorbar off;

i = 17;
subplot(2, 4, 5);
img = imread( sprintf('train/imgs/train%05d.jpg', i) );
imshow(img);
subplot(2, 4, 6);
f = train.X_hog(i, :);
f = reshape(f, 13, 13, 32); % Those dimensions correspond to hog( single(img)/255, 17, 8);
im( hogDraw(f) ); colormap gray;
axis off; colorbar off;

i = 26;
subplot(2, 4, 7);
img = imread( sprintf('train/imgs/train%05d.jpg', i) );
imshow(img);
subplot(2, 4, 8);
f = train.X_hog(i, :);
f = reshape(f, 13, 13, 32); % Those dimensions correspond to hog( single(img)/255, 17, 8);
im( hogDraw(f) ); colormap gray;
axis off; colorbar off;



%% DATA DISTRIBUTION

names{1} = 'plane';
names{2} = 'car';
names{3} = 'horse';
names{4} = 'other';
values.plane = 1;
values.car = 2;
values.horse = 3;
values.other = 4;

labels = categorical(train.y, 1:4, names);

figure;
histogram(labels, 'Normalization', 'probability');
ylabel('probability');












