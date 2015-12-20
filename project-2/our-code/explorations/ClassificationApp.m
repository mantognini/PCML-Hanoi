
%% Generate Data For Classification Application

% Load features and labels of training data
clear
addpath(genpath('toolboxs/'));
addpath(genpath('our-code/'));
load 'data/train/train.mat';

%% Apply PCA
%

fprintf('Splitting into train/test & normalising...\n');
tic

Tr = [];
Te = [];

ratio = 0.7;

idx = randperm(size(train.X_cnn,1));
mid = floor(length(idx) * ratio);

Xtmp = sparse(double(train.X_cnn));

Tr.idxs = idx(1:mid);
Tr.X = Xtmp(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.idxs = idx(mid+1:end);
Te.X = Xtmp(Te.idxs,:);
Te.y = train.y(Te.idxs);

[Tr.normX, mu, sigma] = zscore(full(Tr.X));
Te.normX = normalize(full(Te.X), mu, sigma);

clear idx mid ratio Xtmp;
toc

% PCA for CNN (huge dimensionality)

fprintf('Extracting data...\n');
tic
X = Tr.normX;    % ON NORMALISED DATA
N = size(X, 1);
D = size(X, 2);
assert(N < D);
toc

% X is so big that computing the whole covariance matrix is not possible.
% See Bishop 12.1.4 for details.
% The main idea being that having N > D will result in D - N + 1
% eigenvalues with value 0 and to find the non-zero eignevalues from a
% different matrix.
fprintf('Computing covariance matrix S...\n');
tic
Xavg = mean(X, 1); % per features
% X2 = sparse(N, D);
% for n = 1:N
%     X2(n, :) = (X(n, :) - Xavg);
% end
X2 = X - ones(N, 1) * Xavg; % no longer sparse...
S = X2 * X2' / N;
toc

fprintf('Computing eigenvalues of S...\n');
tic
[V, l] = eig(S, 'vector');
[l, I] = sort(l, 1, 'descend');
V = V(:, I);
% Probably because of imprecision, a handful of value 
% are slightly negative (e.g. -1.6e-16).
l = abs(l);
toc

clear I;

% Compute U's, i.e. eigenvector for X and not X2
fprintf('Computing eigenvalues for original huge matrix...\n');
tic
% The following is too slow...
% U = zeros(D, N);
% for i = 1:N
%     U(:, i) = 1 / sqrt(N * l(i)) * X2' * V(:, i);
%     i
% end
norm = sqrt(N * l)';
norm = repmat(norm, D, 1);
U = X2' * V ./ norm;
toc

%% For classification application
%

% category = 1;
M = 150;
fprintf('transforming data (Z)...\n');
tic
Tr.normZ = Tr.normX * U(:, 1:M);
Te.normZ = Te.normX * U(:, 1:M);
% TrData = [double(toBinary(Tr.y, category)), Tr.normZ];
TrData = [double(Tr.y), Tr.normZ];
toc


%%
% Train model using application "Classification Learner" ...
model = svmMulticlass(TrData);


%% Apply classifier on testing data
%
yPred = model.predictFcn(Te.normZ);
ber = BER(Te.y, yPred)

%%
% Apply our own binary SVM

try1 = toBinary(Tr.y, 1);
try2 = toBinary(Tr.y, 2);
try3 = toBinary(Tr.y, 3);

% WARNING: prediction on training data itself...
genericClassifier = @(y, C, gamma) svmF(Tr.normZ, y, Tr.normZ, @rbfKernel, C, gamma);

% C and gamma where empirically found
yScore1 = genericClassifier(try1, 7, 0.0003);
yScore2 = genericClassifier(try2, 1, 3.5e-4);
yScore3 = genericClassifier(try3, 10, 1e-4);

TrData2 = [double(Tr.y) yScore1 yScore2 yScore3];
