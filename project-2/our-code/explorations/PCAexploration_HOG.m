clearvars;

% Load features and labels of training data
addpath(genpath('data/train/'));
load 'data/train/train.mat';

%%
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

ratio = 0.7;

idx = randperm(size(train.X_hog,1));
mid = floor(length(idx) * ratio);

Tr.idxs = idx(1:mid);
Tr.X = train.X_hog(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.idxs = idx(mid+1:end);
Te.X = train.X_hog(Te.idxs,:);
Te.y = train.y(Te.idxs);

clear idx mid ratio;

%% PCA for HOG ("small" dimensionality)
tic
X = Tr.X;
N = size(X, 1);
D = size(X, 2);
S = cov(X, 1);   % covariance matrix normalised by the number of samples N
% the `1` above define the normalisation factor to 1/N

%[V, D] = eig(S); % NOT SORTED!!!
%[V, D] = eigs(S, size(S, 1) - 1); % EIGS does not support single precision inputs. !!!!

% Manually sort eigenvectors
[U, l] = eig(S, 'vector'); % eigenvector and eigenvalues
[l, I] = sort(l, 1, 'descend');
U = U(:, I);
clear I;
toc

%% Apply PCA:
% compute z_ni and b_i from Bishop 12.10, p. 563

% % THIS IS UTTERLY SLOW!
% tic
% z1 = zeros(N, D);
% for n = 1:N
%     for j = 1:D
%         z1(n, j) = X(n, :) * U(:, j);
%         %          (1xD) x (Dx1)
%     end
%     n
% end
% toc
%
% % This is a bit better: ~30sec
% tic
% z2 = zeros(N, D);
% for n = 1:N
%     z2(n, :) = X(n, :) * U;
% end
% toc
%
% % This is GOOD: ~2sec
% tic
z = X * U;
% toc

Xavg = mean(X, 1); % per features
b = Xavg * U;

%%
% b and z are used to compute the "reduced" version of X given M dimension
% To find the proper M, we compute the distortion measure J in terms of M

Jm = zeros(D, 1);
for m = 1:D
    Jm(m) = sum(l(m+1:D));
end

figure('Name', 'Distortion Measure');
plot(Jm, 'LineWidth', 4);
xlabel('M');
ylabel('J');
title('distortion J vs M');

clear Jm m;

%%
% M = 1000 seems a good upper bound value but something smaller might
% decently work too.

M = 1000;
% Xt = zeros(N, D);
% for n = 1:N
%     Xt(n, :) = z(n, 1:M) * U(:, 1:M)' + b(M+1:D) * U(:, M+1:D)';
% end
% Xtz = z(:, 1:M) * U(:, 1:M)';
% Xtb = b(M+1:D) * U(:, M+1:D)';
% Xt = Xtz + ones(N, 1) * Xtb;
Xt = z(:, 1:M) * U(:, 1:M)' + ones(N, 1) * b(M+1:D) * U(:, M+1:D)';


%% Display lose of information for a given image
addpath(genpath('toolboxs/piotr/'));
for i=1:10
    clf();
    idx = Tr.idxs(i);
    img = imread( sprintf('train/imgs/train%05d.jpg', idx) );

    % show img
    subplot(221);
    imshow(img);
    title(sprintf('%d-th image; Label %d', idx, Tr.y(i)));

    % show hog features analysis
    h = subplot(223);
    Forig = Tr.X(i, :);
    Forig = reshape(Forig, 13, 13, 32); % Those dimensions correspond to hog( single(img)/255, 17, 8);
    im( hogDraw(Forig) ); colormap(h, 'gray');
    title('original HOG features');
    axis off; %colorbar off;
    caxis([0 6]);

    h = subplot(224);
    Freduced = Xt(i, :);
    Freduced = reshape(Freduced, 13, 13, 32);
    im( hogDraw(Freduced) ); colormap(h, 'gray');
    title('reduced HOG features');
    axis off; %colorbar off;
    caxis([0 6]);

    % show lose of information
    h = subplot(222);
    imagesc(hogDraw(Forig) - hogDraw(Freduced));
    title(['M = ' num2str(M)]); colormap(h, 'default');
    axis off; colorbar;
    caxis([-0.1 0.1]);
    
    pause;
end

clear h i img Forig Freduced;


%% NN from z & b
% 

% NOTE: to run me, execute first the three top sections of this file.

fprintf('Training simple neural network..\n');
addpath(genpath('toolboxs/DeepLearnToolbox-master/'));

% convert X to subspace of size M
M = D;%1000;
Tr.Z = Tr.X * U(:, 1:M);
Te.Z = Te.X * U(:, 1:M);

% NOTE: this ain't working... for some reason, "rotating" the space X to Z
%       using U results in 75% of error when using NN while we have 25%
%       of error when re-rotating Z to X with U' -- even when M = D so that
%       we lose only on the precision of the computation.

% err = sort(sort(abs(Tr.X - Tr.Z * U(:, 1:M)'))');
% m = max(max(err));
% imagesc(err); colorbar; caxis([0 m]);

% Setup NN.
inputSize  = M;
innerSize  = 100;
outputSize = 4;
rng(8339);
nn = nnsetup([inputSize innerSize outputSize]);

opts.numepochs  = 15;
opts.batchsize  = 100;
opts.plot       = 1;
nn.learningRate = 2;

% normalize data
[Tr.normZ, mu, sigma] = zscore(Tr.Z);
Te.normZ = normalize(Te.Z, mu, sigma);
[Tr.normZU, mu, sigma] = zscore(Tr.Z * U(:, 1:M)'); % equivalent to Tr.X
Te.normZU = normalize(Te.Z * U(:, 1:M)', mu, sigma);
[Tr.normX, mu, sigma] = zscore(Tr.X);
Te.normX = normalize(Te.X, mu, sigma);

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
[nn, ~] = nntrain(nn, Tr.normZ, LL, opts);

% to get the scores we need to do nnff (feed-forward)
nn.testing = 1;
nn = nnff(nn, Te.normZ, zeros(size(Te.normZ, 1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPredZ = nn.a{end};

% get the most likely class
[~, predictionsZ] = max(nnPredZ, [], 2);

% Apply the same NN with different inputs: ZU
nn = nnsetup([inputSize innerSize outputSize]);
nn.learningRate = 2;
[nn, ~] = nntrain(nn, Tr.normZU, LL, opts);
nn.testing = 1;
nn = nnff(nn, Te.normZU, zeros(size(Te.normZU, 1), nn.size(end)));
nn.testing = 0;
nnPredZU = nn.a{end};
% get the most likely class
[~, predictionsZU] = max(nnPredZU, [], 2);

% Apply the same NN with different inputs: X
nn = nnsetup([inputSize innerSize outputSize]);
nn.learningRate = 2;
[nn, ~] = nntrain(nn, Tr.normX, LL, opts);
nn.testing = 1;
nn = nnff(nn, Te.normX, zeros(size(Te.normX, 1), nn.size(end)));
nn.testing = 0;
nnPredX = nn.a{end};
% get the most likely class
[~, predictionsX] = max(nnPredX, [], 2);

berErrZ = BER(Te.y, predictionsZ);
berErrZU = BER(Te.y, predictionsZU);
berErrX = BER(Te.y, predictionsX);
fprintf('\nBER Testing error  Z: %.2f%%\n', berErrZ * 100);
fprintf('\nBER Testing error ZU: %.2f%%\n', berErrZU * 100);
fprintf('\nBER Testing error  X: %.2f%%\n', berErrX * 100);

figure; 
subplot(131);
imagesc(nnPredZ); colorbar;
title('Z');
subplot(132);
imagesc(nnPredZU); colorbar;
title('ZU');
subplot(133);
imagesc(nnPredX); colorbar;
title('X');

% NOTE: each subplot represent the same information but computed from a
%       different input: each row is a sample of the test set, each
%       column corresponds to a classification label and the color
%       represents the probability of being in this or that category.
