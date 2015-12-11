%% Exploration of Matlab's Statistics and Machine Learning Toolbox
% There's two main part in this exploration script:
%  a) using fitcsvm, with fminsearch, on artificial data
%  b) using fitcecoc on our data, HOG / CNN on multiclass problem
addpath(genpath('our-code/'));

%%
% http://uk.mathworks.com/help/stats/support-vector-machines-svm.html#bsr5o1q
clearvars;
rng('default')
grnpop = mvnrnd([1,0],eye(2),10);
redpop = mvnrnd([0,1],eye(2),10);

N = 500; % per population
redpts = zeros(N,2);
grnpts = redpts;
for i = 1:N
    grnpts(i,:) = mvnrnd(grnpop(randi(10),:),eye(2)*0.2);
    redpts(i,:) = mvnrnd(redpop(randi(10),:),eye(2)*0.2);
end
N = 2 * N;

%%
figure
plot(grnpts(:,1),grnpts(:,2),'go')
hold on
plot(redpts(:,1),redpts(:,2),'ro')
hold off

%%
cdata = [grnpts;redpts];
grp = ones(N,1);
% Green label 1, red label -1
grp(ceil(N/2+0.5):N) = -1;

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(cdata(:,1)):d:max(cdata(:,1)),...
    min(cdata(:,2)):d:max(cdata(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

%%
% Train the classifier
SVMModel = fitcsvm(cdata,grp,'KernelFunction','rbf','ClassNames',[-1 1]);

% Predict scores over the grid
[~,scores] = predict(SVMModel,xGrid);

%%
% Plot the data and the decision boundary
figure;
h(1:2) = gscatter(cdata(:,1),cdata(:,2),grp,'rg','+*');
hold on
h(3) = plot(cdata(SVMModel.IsSupportVector,1),...
    cdata(SVMModel.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'},'Location','Southeast');
axis equal
hold off

%%
% Set up a partition for cross validation. This step causes the cross 
% validation to be fixed. Without this step, the cross validation is 
% random, so a minimization procedure can find a spurious local minimum.
c = cvpartition(N,'KFold',10);

% NOTE: z=[rbf_sigma,boxconstraint]
minfn = @(z)kfoldLoss(fitcsvm(cdata,grp,'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));

opts = optimset('TolX',5e-4,'TolFun',5e-4);
% [searchmin fval] = fminsearch(minfn,randn(2,1),opts) % many local minima
% are possible so we do it multiple times:
m = 5;%20;
fval = zeros(m,1);
z = zeros(m,2);
for j = 1:m;
    fprintf(['searching for best parameter z... ' num2str(j) '/' num2str(m) '\n']);
    tic
    [searchmin, fval(j)] = fminsearch(minfn,randn(2,1),opts);
    z(j,:) = exp(searchmin);
    toc
end

z = z(fval == min(fval),:);

% Use best z to train SVM
SVMModel = fitcsvm(cdata,grp,'KernelFunction','rbf',...
    'KernelScale',z(1),'BoxConstraint',z(2));
[~,scores] = predict(SVMModel,xGrid);


%%
% Generate and classify some new data points:
grnobj = gmdistribution(grnpop,.2*eye(2));
redobj = gmdistribution(redpop,.2*eye(2));

O = 100;
newData = random(grnobj,O);
newData = [newData;random(redobj,O)];
grpData = ones(2 * O,1);
grpData(O+1:2*O) = -1; % red = -1

v = predict(SVMModel,newData);
mydiff = (v == grpData); % Classified correctly


%%
h = nan(7,1);
figure;
h(1:2) = gscatter(cdata(:,1),cdata(:,2),grp,'rg','+*');
hold on
h(3:4) = gscatter(newData(:,1),newData(:,2),v,'mc','**');
h(5) = plot(cdata(SVMModel.IsSupportVector,1),...
    cdata(SVMModel.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h(1:5),{'-1 (training)','+1 (training)','-1 (classified)',...
    '+1 (classified)','Support Vectors'},'Location','Southeast');
axis equal

for ii = mydiff % Plot red circles around correct pts
    h(6) = plot(newData(ii,1),newData(ii,2),'ro','MarkerSize',12);
end

for ii = not(mydiff) % Plot black circles around incorrect pts
    h(7) = plot(newData(ii,1),newData(ii,2),'ko','MarkerSize',12);
end
legend(h,{'-1 (training)','+1 (training)','-1 (classified)',...
    '+1 (classified)','Support Vectors','Correctly Classified',...
    'Misclassified'},'Location','Southeast');
hold off


%%
ber = BER(grpData, v)

%%
% What about our data?
clearvars;
addpath(genpath('data/'));
addpath(genpath('our-code/'));
load 'data/data.mat';


%%
% Split the data

ratio = 0.7; % training-testing sets ratio

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
val.y = data.yTrain(idxValid); % valid y are 4-class

%% APPLY SVM CLASSIFICATION ON HOG FEATURES
M = 100;
[TrZ, TeZ] = pcaHog(M, tr, val.X);

fprintf('learning...\n');
tic
t = templateSVM();%'Verbose', 1);%, ...
%                'SaveSupportVectors', true);
% default: IterationLimit  = 1e6, GapTolerance = 1e-2, KernelFunction = ???
%          Solver = SMO
opts = statset('UseParallel', 1);
MdlSV = fitcecoc(TrZ, tr.y, 'Learners', t, 'Options', opts);
% default: Kfold = 10
isLoss = resubLoss(MdlSV)
toc
% -> w/ M =   3: 200s
% -> w/ M =   5: 220s (with verbose = 1)
% -> w/ M =  50: 80s (with 8 parallel workers)
% -> w/ M = 100: 89s (with 8 parallel workers)

% Determine the amount of disk space that the ECOC model consumes.
infoMdlSV = whos('MdlSV');
mbMdlSV = infoMdlSV.bytes / 1.049e6
% -> w/ M =   2: 0.18MB
%             3: 0.11MB
%             5: 0.14MB
%            50: 0.87MB
%           100: 1.67MB

% Discard the support vectors and related parameters from the trained ECOC
% model. Then, discard the training data from the resulting model by 
% using compact.
Mdl = discardSupportVectors(MdlSV);
CMdl = compact(Mdl);
info = whos('Mdl', 'CMdl');
[bytesCMdl, bytesMdl] = info.bytes;
memReduction = 1 - [bytesMdl bytesCMdl] / infoMdlSV.bytes
% -> w/ M =   2: [0.48 0.88] percent memory reduction
%             3: [0    0.80]
%             5: [0    0.85]
%            50: [0    0.97]
%           100: [0    0.98]

% Apply compact model on testing data
TeyPred = predict(CMdl, TeZ);
ber = BER(val.y, TeyPred)
% -> w/ M =   2: 0.49
%             3: 0.49
%             5: 0.47
%            50: 0.28
%           100: 0.28
oosLoss = loss(CMdl, TeZ, val.y)
% -> w/ M =   2: 0.44
%             3: 0.44
%             5: 0.43
%            50: 0.28
%           100: 0.28

%% APPLY SVM CLASSIFICATION ON CNN FEATURES
M = 750;
[TrZ, TeZ] = pcaCnn(M, tr, val.X);

%%
fprintf('learning...\n');
tic
t = templateSVM('Solver', 'ISDA');
%   BASELINE    'Solver', 'SMO');  % <- default
%   A BIT WORSE 'Solver', 'L1QP'); % <- slower and not better
%   GOOD TOO    'Solver', 'ISDA'); % <- somewhat faster and good results
%   NOT GOOD:   'KernelFunction', 'rbf');
%   NOT GOOD:   'KernelFunction', 'polynomial', 'PolynomialOrder', 3);
%               'Verbose', 1);
%               'SaveSupportVectors', true); % <- use default
% default: IterationLimit  = 1e6, GapTolerance = 1e-2, 
%          KernelFunction = ???
%          Solver = SMO (others are: ISDA, L1QP)
opts = statset('UseParallel', 1);
MdlSV = fitcecoc(TrZ, tr.y, 'Learners', t, 'Options', opts);
% default: Kfold = 10
isLoss = resubLoss(MdlSV)
toc
% -> w/ M =    3: 13s (with 8 parallel workers)
% -> w/ M =    5: 34s (with 8 parallel workers)
% -> w/ M =   50: 60s (with 8 parallel workers)
% -> w/ M =  100: 65s (with 8 parallel workers)
% -> w/ M =  100: 33s (with 8 parallel workers) using ISDA
% -> w/ M =  100: 92s (with 8 parallel workers) using L1QP
% -> w/ M =  200: 65s (with 8 parallel workers)
% -> w/ M =  500: 10s (with 8 parallel workers) <- !!! SUPER FAST BUT WHY?!
% -> w/ M =  750:  4s (with 8 parallel workers) <- !!! SUPER FAST BUT WHY?!
% -> w/ M = 1000:  5s (with 8 parallel workers) <- !!! SUPER FAST BUT WHY?!
% -> w/ M = 2500: 10s (with 8 parallel workers) <- !!! SUPER FAST BUT WHY?!
% NOTE: here it's super fast for some mystic reason but extracting
%       eigenvectors takes quite some time...
%       e.g. M = 100  ->  16s
%            M = 750  ->  70s
%            M = 1000 -> 100s
%            M = 2500 -> 220s
%       so M = 100 is better for time, and close in the results
%       => probably optimal size result-wise is around 750

% Determine the amount of disk space that the ECOC model consumes.
infoMdlSV = whos('MdlSV');
mbMdlSV = infoMdlSV.bytes / 1.049e6
% -> w/ M =   3:  0.16MB
%             5:  0.22MB
%            50:  1.67MB
%           100:  3.28MB
%           200:  6.49MB
%           500: 16.13MB
%           750: 24.16MB
%          1000: 32.19MB
%          2500: 80.39MB

% (In practive we might not even need to compress the model...)
% Discard the support vectors and related parameters from the trained ECOC
% model. Then, discard the training data from the resulting model by 
% using compact.
Mdl = discardSupportVectors(MdlSV);
CMdl = compact(Mdl);
info = whos('Mdl', 'CMdl');
[bytesCMdl, bytesMdl] = info.bytes;
memReduction = 1 - [bytesMdl bytesCMdl] / infoMdlSV.bytes
% -> w/ M =   3: [0    0.86] percent memory reduction
%             5: [0    0.90]
%            50: [0    0.98]
%           100: [0    0.99]
%           200: [0    0.99]
%           500: [0    0.99]
% ...

% Apply compact model on testing data
TeyPred = predict(CMdl, TeZ);
ber = BER(val.y, TeyPred)
% -> w/ M =   3: 0.15
%             5: 0.13
%            50: 0.0941
%           100: 0.0901
%           100: 0.0916 using ISDA solver
%           100: 0.0945 using L1QP solver
%           200: 0.0979
%           500: 0.0987
%           750: 0.0958
%           750: 0.0877 using ISDA solver
%           750: 0.0813 using ISDA solver (on different split)
%          1000: 0.0849
%          2500: 0.1007 <- worse

oosLoss = loss(CMdl, TeZ, val.y)
% -> w/ M =   3: 0.14
%             5: 0.13
%            50: 0.0908
%           100: 0.0885
%           100: 0.0970 using ISDA solver
%           100: 0.0986 using L1QP solver
%           200: 0.0952
%           500: 0.0946
%           750: 0.0923
%           750: 0.0903 using ISDA solver
%           750: 0.0853 using ISDA solver (on different split)
%          1000: 0.0869
%          2500: 0.0959


