%% Matlab's boosting features: fitensemble
% http://uk.mathworks.com/help/stats/fitensemble.html

clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
load 'data/data.mat';


%%
% Split the data

ratio1 = 0.5; % from all data -> training1-rest sets ratio
ratio2 = 0.5; % from rest -> training2-testing sets ratio
% in effect, testing set is 25% of all data
% training1 is used for SVM features extractions (binary)
% training2 is used for ???
% testing is always used for validation (BER on independent data)

N1 = size(data.yTrain, 1);
splitIdx = floor(N1 * ratio1);

idx = randperm(N1);
idxTrain = idx(1:splitIdx);
idxValid = idx(splitIdx + 1:end);

tr1.X.hog = data.hog.train.X(idxTrain, :);
tr1.X.cnn = data.cnn.train.X(idxTrain, :);
tr1.y = data.yTrain(idxTrain); % train y are 4-class

rest.X.hog = data.hog.train.X(idxValid, :);
rest.X.cnn = data.cnn.train.X(idxValid, :);
rest.y = data.yTrain(idxValid); % valid y are 4-class

% split validation into two dataset again
N2 = size(rest.y, 1);
splitIdx = floor(N2 * ratio2);
idx = randperm(N2);
idxTrain = idx(1:splitIdx);
idxValid = idx(splitIdx + 1:end);

tr2.X.hog = rest.X.hog(idxTrain, :);
tr2.X.cnn = rest.X.cnn(idxTrain, :);
tr2.y = rest.y(idxTrain); % train y are 4-class

val.X.hog = rest.X.hog(idxValid, :);
val.X.cnn = rest.X.cnn(idxValid, :);
val.y = rest.y(idxValid); % valid y are 4-class

%%
% Make some predictions for each class:
%  - either class {1} or {2,3,4}
%  - either class {2} or {1,3,4}
%  - either class {3} or {1,2,4}

%%
% fprintf('class 1...\n');
% tic
% %     y1Pred = rbfSvmHogCV2p(tr1, tr2.X, 1);
% %     ber1 = BER(toBinary(tr2.y, 1), y1Pred)
%     % C*     = 9
%     % gamma* = 1.9e-3
%     % BER    = 0.1470
%     
%     % Equivalent:
% %     y1Pred = svmF2(tr1.X.hog, toBinary(tr1.y, 1), tr2.X.hog, @rbfKernel, 9, 1.9e-3);
% %     ber1 = BER(toBinary(tr2.y, 1), y1Pred)
% 
%     % Generate some features...
%     yf1 = svmF(tr1.X.hog, toBinary(tr1.y, 1), tr2.X.hog, @rbfKernel, 9, 1.9e-3);
%     
%     % For testing purposes...
% %     y1Pred   = yf1;
% %     otherIdx = (y1Pred < 0);
% %     y1Pred( otherIdx) = 0;
% %     y1Pred(~otherIdx) = 1;
% %     ber1 = BER(toBinary(tr2.y, 1), y1Pred)
% toc
% pause(0.1);

%%
fprintf('learning for 1...\n');
tic
t = templateSVM('Solver', 'ISDA', ...
                'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
                'Verbose', 1);
opts = statset('UseParallel', 1);
MdlSV1 = fitcecoc(tr1.X.hog, toBinary(tr1.y, 1), 'Learners', t, 'Options', opts);
[y1Pred, yf1] = predict(MdlSV1, tr2.X.hog);
[~, valyf1] = predict(MdlSV1, val.X.hog);
ber1 = BER(toBinary(tr2.y, 1), y1Pred)
% -> 0.16

%%
% fprintf('class 2...\n');
% tic
% %     y2Pred = rbfSvmHogCV2p(tr1, tr2.X, 2);
% %     ber2 = BER(toBinary(tr2.y, 2), y2Pred)
%     % C*     = 11
%     % gamma* = 2e-3
%     % BER    = 0.1220
%     
%     % Generate some features...
%     yf2 = svmF(tr1.X.hog, toBinary(tr1.y, 2), tr2.X.hog, @rbfKernel, 11, 2e-3);
%     
%     % For testing purposes...
% %     y2Pred   = yf2;
% %     otherIdx = (y2Pred < 0);
% %     y2Pred( otherIdx) = 0;
% %     y2Pred(~otherIdx) = 1;
% %     ber2 = BER(toBinary(tr2.y, 2), y2Pred)
% toc
% pause(0.1);

%%
fprintf('learning for 2...\n');
tic
t = templateSVM('Solver', 'ISDA', ...
                'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
                'Verbose', 1);
opts = statset('UseParallel', 1);
MdlSV2 = fitcecoc(tr1.X.hog, toBinary(tr1.y, 2), 'Learners', t, 'Options', opts);
[y2Pred, yf2] = predict(MdlSV2, tr2.X.hog);
[~, valyf2] = predict(MdlSV2, val.X.hog);
ber2 = BER(toBinary(tr2.y, 2), y2Pred)
% -> 0.12

%%
% fprintf('class 3...\n');
% tic
%     y3Pred = rbfSvmHogCV2p(tr1, val.X, 3);
%     ber3 = BER(toBinary(tr2.y, 3), y3Pred)
% toc
% pause(0.1);


%%
fprintf('learning for 3...\n');
tic
t = templateSVM('Solver', 'ISDA', ...
                'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
                'Verbose', 1);
opts = statset('UseParallel', 1);
MdlSV3 = fitcecoc(tr1.X.hog, toBinary(tr1.y, 3), 'Learners', t, 'Options', opts);
[y3Pred, yf3] = predict(MdlSV3, tr2.X.hog);
[~, valyf3] = predict(MdlSV3, val.X.hog);
ber3 = BER(toBinary(tr2.y, 3), y3Pred)
% -> 0.21


%% Train a decision tree ensemble using AdaBoost

% figure; histogram(yf1); hold on; histogram(yf2); histogram(yf3); hold off;
features = [yf1, yf2, yf3];
finalData = tr2.y;

tic
ClassTreeEns = fitensemble(features, finalData, 'AdaBoostM2', 200, 'Tree');
% -> BER = 0.24
% ClassTreeEns = fitensemble(features, finalData, 'LPBoost', 80, 'Tree');
% -> BER = 0.73 XXXX :-( XXXX
% ClassTreeEns = fitensemble(features, finalData, 'TotalBoost', 50, 'Tree');
% -> BER = 0.60 XXXX :-( XXXX
% ClassTreeEns = fitensemble(features, finalData, 'RUSBoost', 200, 'Tree');
% -> BER = 0.36

% NOTE: LPBoost and TotalBoost are supposed to be optimised versions...
%       Thank you Matlab!
toc

% Determine the cumulative resubstitution losses
figure;
rsLoss = resubLoss(ClassTreeEns, 'Mode', 'Cumulative');
plot(rsLoss);
xlabel('Number of Learning Cycles');
ylabel('Resubstitution Loss');

%
valdata = [valyf1, valyf2, valyf3];
yPred = predict(ClassTreeEns, valdata);
ber = BER(val.y, yPred)
% with AdaBoostM2 -> BER = 0.24

%% Baseline: use multiclass classifier without boosting

t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Verbose', 1);
opts = statset('UseParallel', 1);
MdlSV = fitcecoc(tr1.X.hog, tr1.y, 'Learners', t, 'Options', opts);
yPred = predict(MdlSV, val.X.hog);
ber = BER(val.y, yPred)
% -> BER = 0.23
