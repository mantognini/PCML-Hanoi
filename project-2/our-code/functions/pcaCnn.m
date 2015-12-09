function [TrZ, TeZ] = pcaCnn(M, train, XValid)
%
% Project data on the M more significant eigenvectors

    fprintf('[pcaCnn] Extracting data & computing eigenvectors...\n');
    % X is so big that computing the whole covariance matrix is not possible.
    % See Bishop 12.1.4 for details.
    % The main idea being that having N > D will result in D - N + 1
    % eigenvalues with value 0 and to find the non-zero eignevalues from a
    % different matrix.
    tic
    % NOTE: data is already normalised
    N = size(train.X.cnn, 1);
    X2 = double(train.X.cnn) - ones(N, 1) * double(train.cnn.mu); % no longer sparse...
    S = X2 * X2' / N;
    [Vm, Dm] = eigs(S, M); % the M largest eigenvectors
    lm = Dm(1:size(Dm,1)+1:end); % or Dm * ones(M, 1)
    toc

    % Compute U's, i.e. eigenvector for X and not X2
    fprintf('[pcaCnn] Computing eigenvalues for original huge matrix...\n');
    tic
    norm = sqrt(N * lm)';
    % Um = tmp ./ norm;
    tmp = X2' * Vm;
    Um = zeros(size(tmp));
    for i = 1:size(tmp, 2)
        Um(:, i) = tmp(:, i) ./ norm(i);
    end
    toc

    fprintf('[pcaCnn] Convert X to subspace of size M...\n');
    tic
    TrZ = train.X.cnn * Um;
    TeZ = XValid.cnn * Um;
    toc
    
end

