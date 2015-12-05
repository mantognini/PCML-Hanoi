clearvars;

% Load features and labels of training data
addpath(genpath('data/train/'));
load 'data/train/train.mat';

%% PCA for CNN (huge dimensionality)

X = sparse(double(train.X_cnn)); % choose an image representation
N = size(X, 1);
D = size(X, 2);
assert(N < D);

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

% Apply PCA:
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

