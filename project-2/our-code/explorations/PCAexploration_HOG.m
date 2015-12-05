clearvars;

% Load features and labels of training data
addpath(genpath('data/train/'));
load 'data/train/train.mat';

%% PCA for HOG ("small" dimensionality)

X = train.X_hog; % choose an image representation
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

% Apply PCA:
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

clear Jm m Xavg l S;

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
    img = imread( sprintf('train/imgs/train%05d.jpg', i) );

    % show img
    subplot(221);
    imshow(img);
    title(sprintf('%d-th image; Label %d', i, train.y(i)));

    % show hog features analysis
    h = subplot(223);
    Forig = train.X_hog(i, :);
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
