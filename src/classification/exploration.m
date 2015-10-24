%% Explore classification data set

%% Load data & some general facts

clear all;
load('HaNoi_classification');

% X_test    is 1500x40
% X_train   is 1500x40
% y_train   is 1500x1  which values are either -1 or 1

N = length(y_train);
D = size(X_train, 2);

% Feature-wise, we have 5 discrete input variables:
discreteFeatures = [ 16, 25, 26, 32, 38 ];
% # of categories:    4   2   4   2   2
% They all have a relatively uniform distribution among categories,
% especially feature 38 which has 50%-50% distribution for both test
% and train data.

% Continuous input
continuousFeatures = setdiff([1:D], discreteFeatures);

% Features 1, 5, 11 and 36 (less cleanly) have a gaussian distribution
% plus a spike outside the gaussian curve's main domain.
keyContinuousFeatures = [1, 5, 11, 36];
% Bivariate distributions show that there might indeed be something about
% this spike. 


%% Display histogram of response
figure('Name', 'Histogram of response');
y_categorical = categorical(y_train); % convert {-1, 1} to categories "-1" and "1"
h = histogram(y_categorical);
h.Normalization = 'probability';
xlabel('output categories');
ylabel('probability');
ylim([0 1]);


%% Display only discrete features
figure('Name', 'Distributions of discrete features (train + test)');
for i = 1:length(discreteFeatures)
    subplot(2, 3, i);
    f = discreteFeatures(i);
    
    plotFeature(true, f, X_train, X_test);
end


%% Display only discrete features with respect to the response
figure('Name', 'Distributions of discrete features with the response');
for i = 1:length(discreteFeatures)
    subplot(2, 3, i);
    f = discreteFeatures(i);
    
    plotFeatureResponse(f, X_train, y_train);
end

% For reference:
subplot(2, 3, 6);
y_categorical = categorical(y_train); % convert {-1, 1} to categories "-1" and "1"
h = histogram(y_categorical);
h.Normalization = 'probability';
xlabel('output categories');
ylabel('probability');
%ylim([0 1]);


%% Display only continuous features
figures = length(continuousFeatures);
figNo = 0;
for i = 1:figures
    if (mod(i - 1, 9) == 0)
        figNo = figNo + 1;
        figure('Name', ['Histograms of continuous features (train + test) ' num2str(figNo) ' / 4']);
    end
    subplot(3, 3, mod(i - 1, 9) + 1);
    f = continuousFeatures(i);
    
    plotFeature(false, f, X_train, X_test);
end


%% Display bivariate histograms of key features with response
figure('Name', 'Histograms of key features with response');
for i = 1:length(keyContinuousFeatures);
    f = keyContinuousFeatures(i);
    
    subplot(2, 2, i);
    
    plotFeatureResponse(f, X_train, y_train);
end


%% Look at feature 1 & 11
clear all;
data = loadClassificationData();

f1  = data.train.X(:, 1);
f11 = data.train.X(:, 11);

y = data.train.y;
idx = y == 1;

figure('Name', 'Histograms of features 1 & 11 with response');
histogram2(f1(idx), f11(idx));
hold on;
histogram2(f1(~idx), f11(~idx));

legend('1', '0', 'Location', 'northeast');

xlabel('1st');
ylabel('11th');
zlabel('count');


%% Plot categories (not really useful actually)

clear all;
data = loadClassificationData();

% Split data into training and validation sets
N = length(data.train.y);
idx = randperm(N);
X = data.train.X(idx, :);
y = data.train.y(idx);

splitRatio = 0.7;
[XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);

y = naiveClassificationMethod(XTr, yTr, XValid);
X = data.train.X(:, [11, 7]); % two random features for 2D display

correctIdx = find(y == yValid);
correct0Idx = find(y(correctIdx) == 0);
correct1Idx = find(y(correctIdx) == 1);
incorrectIdx = find(y ~= yValid);

assert(~isempty(correct0Idx), 'at least one 0 correctly classified');
assert(~isempty(correct1Idx), 'at least one 1 correctly classified');
assert(~isempty(incorrectIdx), 'at least one point misclassified');

figure();
plot(X(correct0Idx, 1), X(correct0Idx, 2), 'bo', ...
     X(correct1Idx, 1), X(correct1Idx, 2), 'ro', ...
     X(incorrectIdx, 1), X(incorrectIdx, 2), 'go');
legend('0', '1', 'inkorect');

