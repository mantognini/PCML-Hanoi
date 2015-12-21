%
% Produce mock testing data
clearvars;
load 'train/train.mat';

% Add batches
N = size(train.y, 1);
test.X_cnn = [];
test.X_hog = [];
y = [];
for i = 1:6
    idx = randperm(floor(N/ 8));

    test.X_cnn = [test.X_cnn; train.X_cnn(idx, :)];
    test.X_hog = [test.X_hog; train.X_hog(idx, :)];
    y = [y; train.y(idx, :)];
end

% Save
save 'mock-test.mat' test;
save 'mock-solution.mat' y;