clearvars;

% Load features and labels of training data
addpath(genpath('data/train/'));
addpath(genpath('toolboxs/DeepLearnToolbox-master/'));
load 'data/train/train.mat';

%%
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

[Tr.normX, mu, sigma] = zscore(Tr.X);
Te.normX = normalize(Te.X, mu, sigma);

clear idx mid ratio Xtmp;
toc

%% PCA for CNN (huge dimensionality)

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


%% Apply PCA:
% compute z_ni and b_i from Bishop 12.10, p. 563
% See PCAexploration_HOG for more details
fprintf('Computing z and b\n');
tic
z = X2 * U;
b = Xavg * U;
toc

%%
% b and z are used to compute the "reduced" version of X given M dimension
% To find the proper M, we compute the distortion measure J in terms of M

% NOTE: we use N in place of D since N < D!
Jm = zeros(N, 1);
for m = 1:N
    Jm(m) = sum(l(m+1:N));
end

figure('Name', 'Distortion Measure');
plot(Jm, 'LineWidth', 4);
xlabel('M');
ylabel('J');
title('distortion J vs M');

clear Jm m;

%%
% Upper bound value for M is not as clearly identifyable as with HOG.
% Probably something around M = 3000 should do the trick, maybe lower
% value works too however.

M = 3000;
% Xt = zeros(N, D);
% for n = 1:N
%     Xt(n, :) = z(n, 1:M) * U(:, 1:M)' + b(M+1:D) * U(:, M+1:D)';
% end
% Xtz = z(:, 1:M) * U(:, 1:M)';
% Xtb = b(M+1:D) * U(:, M+1:D)';
% Xt = Xtz + ones(N, 1) * Xtb;
fprintf('Computing X~ (reduced X)...\n');
tic
% NOTE: we use N in place of D since N < D!
Xt = z(:, 1:M) * U(:, 1:M)' + ones(N, 1) * b(M+1:N) * U(:, M+1:N)';
toc


%% Display lose of information for a given image

% show lose of information, regroup error of the same magnitude together
fprintf('Preparing data...\n');
tic
diff = sort(abs(X - Xt));
diff = sort(diff');
diff = log(diff);
upperBound = max(max(diff));
toc

imagesc(diff);
title(['M = ' num2str(M)]); colormap 'default';
colorbar;
caxis([0 upperBound]);

%% NN from U
% 

% NOTE: to run me, execute first the three top sections of this file.

fprintf('Training simple neural network...\n');

% convert X to subspace of size M
M = 500; % 500 seems to works decently well
fprintf('transforming data (Z)...\n');
tic
Tr.normZ = Tr.normX * U(:, 1:M);
Te.normZ = Te.normX * U(:, 1:M);
toc

% Setup NN.
inputSize  = M;
innerSize  = 50;
outputSize = 4;
%rng(8339);
nn = nnsetup([inputSize innerSize outputSize]);

opts.numepochs  = 15;
opts.batchsize  = 100;
opts.plot       = 1;
nn.learningRate = 2;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor(size(Tr.normZ) / opts.batchsize);
Tr.normZ     = Tr.normZ(1:numSampToUse, :);
labels       = Tr.y(1:numSampToUse);

% prepare labels for NN
LL = [1 * (labels == 1), ... % first column, p(y=1)
      1 * (labels == 2), ... % second column, p(y=2), etc
      1 * (labels == 3), ...
      1 * (labels == 4) ];
  
  
fprintf('training...\n');
tic
[nn, ~] = nntrain(nn, Tr.normZ, LL, opts);
toc

% to get the scores we need to do nnff (feed-forward)
fprintf('testing...\n');
tic
nn.testing = 1;
nn = nnff(nn, Te.normZ, zeros(size(Te.normZ, 1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPredZ = nn.a{end};

% get the most likely class
[~, predictionsZ] = max(nnPredZ, [], 2);
toc

% Apply the same NN with different inputs: ZU
% NOTE: not doing this as the matrixes are HUGE!!!
% fprintf('transforming data (ZU)...\n');
% tic
% Tr.normZU = Tr.normX * U(:, 1:M) * U(:, 1:M)';
% Te.normZU = Te.normX * U(:, 1:M) * U(:, 1:M)';
% toc
% inputSize = size(Tr.normZU, 2);
% nn = nnsetup([inputSize innerSize outputSize]);
% nn.learningRate = 2;
% [nn, ~] = nntrain(nn, Tr.normZU, LL, opts);
% nn.testing = 1;
% nn = nnff(nn, Te.normZU, zeros(size(Te.normZU, 1), nn.size(end)));
% nn.testing = 0;
% nnPredZU = nn.a{end};
% % get the most likely class
% [~, predictionsZU] = max(nnPredZU, [], 2);

% Apply the same NN with different inputs: X
fprintf('Applaying NN on original data...\n');
tic
inputSize = size(Tr.X, 2); % NOT Tr.normX as it is HUGE and NOT sparse
nn = nnsetup([inputSize innerSize outputSize]);
nn.learningRate = 2;
[nn, ~] = nntrain(nn, Tr.X, LL, opts);
nn.testing = 1;
nn = nnff(nn, Te.X, zeros(size(Te.X, 1), nn.size(end)));
nn.testing = 0;
nnPredX = nn.a{end};
% get the most likely class
[~, predictionsX] = max(nnPredX, [], 2);
toc

berErrZ = BER(Te.y, predictionsZ);
% berErrZU = BER(Te.y, predictionsZU);
berErrX = BER(Te.y, predictionsX);
fprintf('\nBER Testing error  Z: %.2f%%\n', berErrZ * 100);
% fprintf('\nBER Testing error ZU: %.2f%%\n', berErrZU * 100);
fprintf('\nBER Testing error  X: %.2f%%\n', berErrX * 100);

figure; 
subplot(121);
imagesc(nnPredZ); colorbar;
title('Z');
% subplot(132);
% imagesc(nnPredZU); colorbar;
% title('ZU');
subplot(122);
imagesc(nnPredX); colorbar;
title('X');

% NOTE: each subplot represent the same information but computed from a
%       different input: each row is a sample of the test set, each
%       column corresponds to a classification label and the color
%       represents the probability of being in this or that category.
