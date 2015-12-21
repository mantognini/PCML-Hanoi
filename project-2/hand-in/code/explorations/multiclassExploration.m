%% Explore multiclass classification with:
% - SVM classifier
% ... and maybe more
% http://uk.mathworks.com/help/stats/fitcecoc.html#input_argument_namevalue_learners
%

clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
addpath(genpath('toolboxs/'));
load 'data/fixedData.mat';
% load 'data/data.mat';
data = fixedData;

names{1} = 'plane';
names{2} = 'car';
names{3} = 'horse';
names{4} = 'other';
values.plane = 1;
values.car = 2;
values.horse = 3;
values.other = 4;


%% SETTINGS

ratio1 = 0.5; % from all data -> training1-rest sets ratio
ratio2 = 0.5; % from rest -> training2-testing sets ratio
% in effect, testing set is 30% of all data
% training1 is used for SVM features extractions
% training2 is used for training the final model
% testing is always used for validation (BER on independent data)

M = 150; % PCA for CNN

% For fitcecoc:
% opts = statset('UseParallel', 1);
opts = statset();

% WHY IS THE DEFAULT ONE WORKING AND THE OTHER NOT?!?
svmDefault = templateSVM('Verbose', 0);

svmPoly = ...
    templateSVM('Solver', 'SMO', ...
                'KernelFunction', 'polynomial', 'PolynomialOrder', 3, ...
                'KernelScale', 'auto', ...
                'Verbose', 1);

svmRbf = ...
    templateSVM('Solver', 'SMO', ...
                'KernelFunction', 'rbf', ...
                'KernelScale', 'auto', ...
                'Verbose', 1);
            
knn = templateKNN('NumNeighbors', 5, 'Standardize', 1);


%% Split the data

N1 = size(data.yTrain, 1);
splitIdx = floor(N1 * ratio1);

idx1 = randperm(N1);
idxTrain = idx1(1:splitIdx);
idxRest = idx1(splitIdx + 1:end);

tr1.X.hog = data.hog.train.X(idxTrain, :);
tr1.X.cnn = data.cnn.train.X(idxTrain, :);
tr1.y = data.yTrain(idxTrain); % train y are 4-class

rest.X.hog = data.hog.train.X(idxRest, :);
rest.X.cnn = data.cnn.train.X(idxRest, :);
rest.y = data.yTrain(idxRest); % valid y are 4-class

% split validation into two dataset again
N2 = size(rest.y, 1);
splitIdx = floor(N2 * ratio2);
idx2 = randperm(N2);
idxTrain2 = idx2(1:splitIdx);
idxValid = idx2(splitIdx + 1:end);

tr2.X.hog = rest.X.hog(idxTrain2, :);
tr2.X.cnn = rest.X.cnn(idxTrain2, :);
tr2.y = rest.y(idxTrain2); % train y are 4-class

val.X.hog = rest.X.hog(idxValid, :);
val.X.cnn = rest.X.cnn(idxValid, :);
val.y = rest.y(idxValid); % valid y are 4-class


%% Apply PCA on CNN

% [TrZ, TeZ] = pcaCnn(M, train, XValid); % can't apply it here...
fprintf('Extracting data & computing eigenvectors...\n');
% X is so big that computing the whole covariance matrix is not possible.
% See Bishop 12.1.4 for details.
% The main idea being that having N > D will result in D - N + 1
% eigenvalues with value 0 and to find the non-zero eignevalues from a
% different matrix.
tic
% NOTE: data is already normalised
Tr1X = double(tr1.X.cnn);
Tr2X = double(tr2.X.cnn);
valX = double(val.X.cnn);
N = size(Tr1X, 1);
S = Tr1X * Tr1X' / N;
[Vm, Dm] = eigs(S, M); % the M largest eigenvectors
lm = Dm(1:size(Dm,1)+1:end); % or Dm * ones(M, 1)
toc

fprintf('Computing eigenvalues for original huge matrix...\n');
tic
norm = sqrt(N * lm)';
% Um = tmp ./ norm;
tmp = Tr1X' * Vm;
Um = zeros(size(tmp));
for i = 1:size(tmp, 2)
    Um(:, i) = tmp(:, i) ./ norm(i);
end
toc

fprintf('Convert X to subspace of size M...\n');
tic
Tr1Z = Tr1X * Um;
Tr2Z = Tr2X * Um;
valZ = valX * Um;
toc


%% Display data

figure;
subplot(221);
histogram(tr1.y, 'Normalization', 'probability'); hold on;
histogram(tr2.y, 'Normalization', 'probability'); hold on;
histogram(val.y, 'Normalization', 'probability'); hold off;
title('multiclass');
subplot(222);
histogram(toBinary(tr1.y, 1), 'Normalization', 'probability'); hold on;
histogram(toBinary(tr2.y, 1), 'Normalization', 'probability'); hold on;
histogram(toBinary(val.y, 1), 'Normalization', 'probability'); hold off;
title('category 1');
subplot(223);
histogram(toBinary(tr1.y, 2), 'Normalization', 'probability'); hold on;
histogram(toBinary(tr2.y, 2), 'Normalization', 'probability'); hold on;
histogram(toBinary(val.y, 2), 'Normalization', 'probability'); hold off;
title('category 2');
subplot(224);
histogram(toBinary(tr1.y, 3), 'Normalization', 'probability'); hold on;
histogram(toBinary(tr2.y, 3), 'Normalization', 'probability'); hold on;
histogram(toBinary(val.y, 3), 'Normalization', 'probability'); hold off;
title('category 3');

pause(0.1);

%% SVM MULTICLASS HOG

fprintf('SVM MULTICLASS HOG...\n');
tic

model = fitcecoc(tr1.X.hog, tr1.y, 'Learners', svmDefault, 'Options', opts);

[trLabelPredSvmHog4, trScoreSvmHog4] = predict(model, tr2.X.hog);
[valLabelPredSvmHog4, valScoreSvmHog4] = predict(model, val.X.hog);

% THIS SHOULD NOT BE DONE IN PRACTICE!!
berSvmHog4 = BER(tr2.y, trLabelPredSvmHog4);

toc
fprintf(['SVM MULTI HOG BER = ' num2str(berSvmHog4) '\n']);


% %%
% figure;
% subplot(211);
% histogram(trLabelPredSvmHog4, 'Normalization', 'probability'); hold on;
% histogram(tr2.y, 'Normalization', 'probability'); hold off;
% subplot(212);
% for cat = 1:4
%     histogram(trScoreSvmHog4(:, cat)); hold on;
% end


%% SVM MULTICLASS CNN

fprintf('SVM MULTICLASS CNN...\n');
tic

model = fitcecoc(Tr1Z, tr1.y, 'Learners', svmDefault, 'Options', opts);

[trLabelPredSvmCnn4, trScoreSvmCnn4] = predict(model, Tr2Z);
[valLabelPredSvmCnn4, valScoreSvmCnn4] = predict(model, valZ);

% THIS SHOULD NOT BE DONE IN PRACTICE!!
berSvmCnn4 = BER(tr2.y, trLabelPredSvmCnn4);

toc
fprintf(['SVM MULTI CNN BER = ' num2str(berSvmCnn4) '\n']);


% %%
% figure;
% subplot(211);
% histogram(trLabelPredSvmCnn4, 'Normalization', 'probability'); hold on;
% histogram(tr2.y, 'Normalization', 'probability'); hold off;
% subplot(212);
% for cat = 1:4
%     histogram(trScoreSvmCnn4(:, cat)); hold on;
% end


%% SVM 4x BINARY HOG

ticId = ticStatus('SVM BINARY HOG');
trScoreSvmHog2 = [];
trLabelPredSvmHog2 = [];
valScoreSvmHog2 = [];
valLabelPredSvmHog2 = [];
berSvmHog2 = zeros(4, 0);
for category = 1:4
    model = fitcecoc(tr1.X.hog, toBinary(tr1.y, category), ...
                     'Learners', svmDefault, 'Options', opts);
                 
    [trLabelPredSvmHog2cat, trScoreSvmHog2cat] = predict(model, tr2.X.hog);
    [valLabelPredSvmHog2cat, valScoreSvmHog2cat] = predict(model, val.X.hog);
    
    % THIS SHOULD NOT BE DONE IN PRACTICE!!
    berSvmHog2(category) = BER(toBinary(tr2.y, category), trLabelPredSvmHog2cat);
    
    % keep track of the predictions for this category
    trLabelPredSvmHog2 = [trLabelPredSvmHog2, trLabelPredSvmHog2cat];
    trScoreSvmHog2 = [trScoreSvmHog2, trScoreSvmHog2cat];
    valLabelPredSvmHog2 = [valLabelPredSvmHog2, valLabelPredSvmHog2cat];
    valScoreSvmHog2 = [valScoreSvmHog2, valScoreSvmHog2cat];
    
    tocStatus(ticId, category/4);
end

fprintf(['SVM BINARY HOG BER = ' num2str(berSvmHog2) '\n']);


% %%
% for category = 0:3
%     figure;
%     subplot(211);
%     histogram(trLabelPredSvmHog2(:, category + 1), 'Normalization', 'probability');
%     subplot(212);
%     histogram(trScoreSvmHog2(:, 2 * category + 1), 'Normalization', 'probability'); hold on;
%     histogram(trScoreSvmHog2(:, 2 * category + 2), 'Normalization', 'probability'); hold off;
% end



%% SVM 4x BINARY CNN

ticId = ticStatus('SVM BINARY CNN');
trScoreSvmCnn2 = [];
trLabelPredSvmCnn2 = [];
valScoreSvmCnn2 = [];
valLabelPredSvmCnn2 = [];
berSvmCnn2 = zeros(4, 0);
berValSvmCnn2 = zeros(4, 0);
for category = 1:4
    model = fitcecoc(Tr1Z, toBinary(tr1.y, category), ...
                     'Learners', svmDefault, 'Options', opts);
                 
    [trLabelPredSvmCnn2cat, trScoreSvmCnn2cat] = predict(model, Tr2Z);
    [valLabelPredSvmCnn2cat, valScoreSvmCnn2cat] = predict(model, valZ);
    
    % THIS SHOULD NOT BE DONE IN PRACTICE!!
    berSvmCnn2(category) = BER(toBinary(tr2.y, category), trLabelPredSvmCnn2cat);
    berValSvmCnn2(category) = BER(toBinary(val.y, category), valLabelPredSvmCnn2cat);
    
    % keep track of the predictions for this category
    trLabelPredSvmCnn2 = [trLabelPredSvmCnn2, trLabelPredSvmCnn2cat];
    trScoreSvmCnn2 = [trScoreSvmCnn2, trScoreSvmCnn2cat];
    valLabelPredSvmCnn2 = [valLabelPredSvmCnn2, valLabelPredSvmCnn2cat];
    valScoreSvmCnn2 = [valScoreSvmCnn2, valScoreSvmCnn2cat];
    
    tocStatus(ticId, category/4);
end

fprintf(['SVM BINARY CNN BER = ' num2str(berSvmCnn2) '\n']);
fprintf(['SVM VAL BINARY CNN BER = ' num2str(berValSvmCnn2) '\n']);


% %%
% for category = 0:3
%     figure;
%     subplot(211);
%     histogram(trLabelPredSvmCnn2(:, category + 1), 'Normalization', 'probability');
%     subplot(212);
%     histogram(trScoreSvmCnn2(:, 2 * category + 1), 'Normalization', 'probability'); hold on;
%     histogram(trScoreSvmCnn2(:, 2 * category + 2), 'Normalization', 'probability'); hold off;
% end


%% HOMEMADE BINARY SVM

tic

try1 = toBinary(tr1.y, 1);
try2 = toBinary(tr1.y, 2);
try3 = toBinary(tr1.y, 3);

genericClassifier = @(X, y, C, gamma) svmF(Tr1Z, y, X, @rbfKernel, C, gamma);

% C and gamma where empirically found
trScoreHomeMadeSvmCnn2 = zeros(size(Tr2Z, 1), 3);
trScoreHomeMadeSvmCnn2(:, 1) = genericClassifier(Tr2Z, try1, 7, 0.0003);
trScoreHomeMadeSvmCnn2(:, 2) = genericClassifier(Tr2Z, try2, 1, 3.5e-4);
trScoreHomeMadeSvmCnn2(:, 3) = genericClassifier(Tr2Z, try3, 10, 1e-4);

valScoreHomeMadeSvmCnn2 = zeros(size(valZ, 1), 3);
valScoreHomeMadeSvmCnn2(:, 1) = genericClassifier(valZ, try1, 7, 0.0003);
valScoreHomeMadeSvmCnn2(:, 2) = genericClassifier(valZ, try2, 1, 3.5e-4);
valScoreHomeMadeSvmCnn2(:, 3) = genericClassifier(valZ, try3, 10, 1e-4);

% trainingData = [double(tr2.y) trScoreHomeMadeSvmCnn2];

toc

% idx1 = tr2.y == 1;
% idx2 = tr2.y == 2;
% idx3 = tr2.y == 3;
% idx4 = tr2.y == 4;
% 
% figure;
% plot3(trScoreHomeMadeSvmCnn2(idx1, 1), ...
%       trScoreHomeMadeSvmCnn2(idx1, 2), ...
%       trScoreHomeMadeSvmCnn2(idx1, 3), ...
%       '.', 'MarkerSize', 30);
% hold on;
% plot3(trScoreHomeMadeSvmCnn2(idx2, 1), ...
%       trScoreHomeMadeSvmCnn2(idx2, 2), ...
%       trScoreHomeMadeSvmCnn2(idx2, 3), ...
%       '.', 'MarkerSize', 30);
% plot3(trScoreHomeMadeSvmCnn2(idx3, 1), ...
%       trScoreHomeMadeSvmCnn2(idx3, 2), ...
%       trScoreHomeMadeSvmCnn2(idx3, 3), ...
%       '.', 'MarkerSize', 30);
% plot3(trScoreHomeMadeSvmCnn2(idx4, 1), ...
%       trScoreHomeMadeSvmCnn2(idx4, 2), ...
%       trScoreHomeMadeSvmCnn2(idx4, 3), ...
%       '.', 'MarkerSize', 30);
% legend('class 1', 'class 2', 'class 3', 'class 4');
% xlabel('pred 1');
% ylabel('pred 2');
% zlabel('pred 3');
% grid on;

%% Trees On HOG

tic

t = templateTree('Surrogate','on');
model = ...%fitensemble(tr1.X.hog, tr1.y, 'AdaBoostM2', 20, t);
    fitcecoc(tr1.X.hog, tr1.y, ...
             'Learners', t, 'Options', opts);

[trLabelPredTreeHog4, trScoreTreeHog4] = predict(model, tr2.X.hog);
[valLabelPredTreeHog4, valScoreTreeHog4] = predict(model, val.X.hog);

% THIS SHOULD NOT BE DONE IN PRACTICE!!
berTreeHog = BER(tr2.y, trLabelPredTreeHog4);
toc
fprintf(['TREE HOG BER = ' num2str(berTreeHog) '\n']);


%% Trees On CNN

tic

t = templateTree('Surrogate','on');
model = ...%fitensemble(Tr1Z, tr1.y, 'AdaBoostM2', 100, t);
    fitcecoc(Tr1Z, tr1.y, ...
             'Learners', t, 'Options', opts);

[trLabelPredTreeCnn4, trScoreTreeCnn4] = predict(model, Tr2Z);
[valLabelPredTreeCnn4, valScoreTreeCnn4] = predict(model, valZ);

% THIS SHOULD NOT BE DONE IN PRACTICE!!
berTreeCnn = BER(tr2.y, trLabelPredTreeCnn4);
toc
fprintf(['TREE CNN BER = ' num2str(berTreeCnn) '\n']);


%% Random forest on HOG

forestOpts.M = 20; % # trees to train
forest = forestTrain(tr1.X.hog, tr1.y, forestOpts);
[trLabelPredForestHog4, trScoreForestHog4] = forestApply(tr2.X.hog, forest);
[valLabelPredForestHog4, valScoreForestHog4] = forestApply(val.X.hog, forest);

% THIS SHOULD NOT BE DONE IN PRACTICE!!
berForestHog = BER(tr2.y, trLabelPredForestHog4);
fprintf(['FOREST HOG BER = ' num2str(berForestHog) '\n']);


%% Random forest on CNN

forestOpts.M = 10; % # trees to train
forest = forestTrain(Tr1Z, tr1.y, forestOpts);
[trLabelPredForestCnn4, trScoreForestCnn4] = forestApply(single(Tr2Z), forest);
[valLabelPredForestCnn4, valScoreForestCnn4] = forestApply(single(valZ), forest);

% THIS SHOULD NOT BE DONE IN PRACTICE!!
berForestCnn = BER(tr2.y, trLabelPredForestCnn4);
fprintf(['FOREST CNN BER = ' num2str(berForestCnn) '\n']);


%% Combine stuff together

% trScores  = [trScoreSvmHog2,  trScoreSvmHog4,  trScoreSvmCnn2,  trScoreSvmCnn4,  trScoreHomeMadeSvmCnn2,  trScoreTreeHog4,  trScoreTreeCnn4,  trLabelPredForestHog4,  trLabelPredForestCnn4];
% valScores = [valScoreSvmHog2, valScoreSvmHog4, valScoreSvmCnn2, valScoreSvmCnn4, valScoreHomeMadeSvmCnn2, valScoreTreeHog4, valScoreTreeCnn4, valLabelPredForestHog4, valLabelPredForestCnn4];
trScores  = [trScoreHomeMadeSvmCnn2];
valScores = [valScoreHomeMadeSvmCnn2];


%% Apply NN

yPredNn = nn(200, 0.05, 500, 0, trScores, tr2.y, valScores);
berNn = BER(val.y, yPredNn);
fprintf(['NN on extracted features: BER = ' num2str(berNn) '\n']);
% Apparently equivalent as using CNN directly...


%% Apply decision tree

t = templateTree('Surrogate','on');
model = fitcecoc(trScores, tr2.y, 'Learners', t, 'Options', opts);
[yPredTree, yScoreTree] = predict(model, valScores);
berTree = BER(val.y, yPredTree);
fprintf(['Tree on extracted features: BER = ' num2str(berTree) '\n']);


%% Apply another kind of decision tree

t = templateTree('Surrogate','on');
model = fitensemble(trScores, tr2.y, 'AdaBoostM2', 100, t);
[yPredTreeBis, yScoreTreeBis] = predict(model, valScores);
berTreeBis = BER(val.y, yPredTreeBis);
fprintf(['Tree (bis) on extracted features: BER = ' num2str(berTreeBis) '\n']);


%% Apply KNN

t = templateKNN('NumNeighbors',5,'Standardize',1);
model = fitensemble(trScores, tr2.y, 'Subspace', 100, t);
[yPredKnn, yScoreKnn] = predict(model, valScores);
berKnn = BER(val.y, yPredKnn);
fprintf(['KNN on extracted features: BER = ' num2str(berKnn) '\n']);


%% Apply Bagging

t = templateTree('Surrogate','on');
model = fitensemble(trScores, tr2.y, 'Bag', 100, t, 'Type', 'Classification');
[yPredBag, yScoreBag] = predict(model, valScores);
berBag = BER(val.y, yPredBag);
fprintf(['Bagging on extracted features: BER = ' num2str(berBag) '\n']);

%% More trees

% leafs = logspace(1,2,20);
% LEAFS = numel(leafs);
% err = zeros(LEAFS,1);
% for n=1:LEAFS
%     t = fitctree(trScores, tr2.y, 'CrossVal', 'On', 'MinLeaf', leafs(n));
%     err(n) = kfoldLoss(t);
% end
% plot(leafs,err);
% xlabel('Min Leaf Size');
% ylabel('cross-validated error');

model = fitctree(trScores, tr2.y, 'MinLeaf', 50);
yPredCtree = predict(model, valScores);
berCtree = BER(val.y, yPredCtree);
fprintf(['CTree on extracted features: BER = ' num2str(berCtree) '\n']);
% view(model,'mode','graph');


%% Piotr' random forest

forestOpts.M = 30; % # trees to train
forest = forestTrain(trScores, tr2.y, forestOpts);
[yPredForest, yScroreForest] = forestApply(single(valScores), forest);
berForest = BER(val.y, yPredForest);
fprintf(['Forest on extracted features: BER = ' num2str(berForest) '\n']);


%%
figure;
subplot(121);
histogram(val.y, 'Normalization', 'probability'); hold on;
histogram(yPredNn, 'Normalization', 'probability'); hold off;
subplot(122);
imagesc(val.y ~= yPredNn);
title(['FINAL BER = ' num2str(berNn)]);


%%

restLabels = rest.y;
restLabels(idxValid) = yPredNn; % override with our prediction
allLabels = data.yTrain;
allLabels(idxRest) = restLabels; % override with our prediction

idxMissclassified = find(allLabels ~= data.yTrain);
fprintf(['# miss classified = ' num2str(length(idxMissclassified)) '\n']);

for cat = 1:4
    idxCat = find(data.yTrain == cat);
    idxMissclassifiedCat = intersect(idxCat, idxMissclassified);
    
    fprintf(['# miss classified cat ' names{cat} ' = ' num2str(length(idxMissclassifiedCat)) '\n']);
    
    figure('Name', ['Category ' names{cat}]);
    
    subplotIdx = 1;
    suplotSize = ceil(sqrt(length(idxMissclassifiedCat)));
    
    for j = 1:length(idxMissclassifiedCat)
        i = idxMissclassifiedCat(j);
        
        subplot(suplotSize, suplotSize, subplotIdx);
        img = imread( sprintf('train/imgs/train%05d.jpg', i) );
        imshow(img);
        title([num2str(i) '-th, p=' names{allLabels(i)}]);
        
        subplotIdx = subplotIdx + 1;
    end
end


