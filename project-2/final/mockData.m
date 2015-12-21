%
% Produce mock testing data
clearvars;
load 'train/train.mat';

% Add batches
N = size(train.y, 1);
test.X_cnn = [];
test.X_hog = [];
y = [];
for i = 1:1
    idx = randperm(N);
    idx = idx([1:floor(N/6)]);

    test.X_cnn = [test.X_cnn; train.X_cnn(idx, :)];
    test.X_hog = [test.X_hog; train.X_hog(idx, :)];
    y = [y; train.y(idx, :)];
end

% Save
save 'mock-test.mat' test;
save 'mock-solution.mat' y;