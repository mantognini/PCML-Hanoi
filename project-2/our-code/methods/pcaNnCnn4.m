function yPred = pcaNnCnn4(train, XValid)
%
% Apply PCA followed by NN on CNN feature

    % SETTINGS:
    M          = 200; % might be non-optimal
    INNER_SIZE = 100;
    EPOCHS     = 30;

    fprintf('Extracting data & computing eigenvectors...\n');
    % X is so big that computing the whole covariance matrix is not possible.
    % See Bishop 12.1.4 for details.
    % The main idea being that having N > D will result in D - N + 1
    % eigenvalues with value 0 and to find the non-zero eignevalues from a
    % different matrix.
    tic
    [TrNormX, mu, sigma] = zscore(double(train.X.cnn));
    TeNormX = normalize(double(XValid.cnn), mu, sigma);
    N = size(TrNormX, 1);
    X2 = TrNormX - ones(N, 1) * mu; % no longer sparse...
    S = X2 * X2' / N;
    [Vm, Dm] = eigs(S, M); % the M largest eigenvectors
    lm = Dm(1:size(Dm,1)+1:end); % or Dm * ones(M, 1)
    toc

    % Compute U's, i.e. eigenvector for X and not X2
    fprintf('Computing eigenvalues for original huge matrix...\n');
    tic
    norm = sqrt(N * lm)';
    % Um = tmp ./ norm;
    tmp = X2' * Vm;
    Um = zeros(size(tmp));
    for i = 1:size(tmp, 2)
        Um(:, i) = tmp(:, i) ./ norm(i);
    end
    toc

    fprintf('convert X to subspace of size M...\n');
    tic
    TrNormZ = TrNormX * Um;
    TeNormZ = TeNormX * Um;
    toc

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 0, TrNormZ, train.y, TeNormZ);
end

