function [TrZ, TeZ] = pcaManual(X, XValid, M)
%
% Project data on the M more significant eigenvectors manually

    fprintf('[pcaManual] Extracting data & computing eigenvectors...\n');
    % NOTE: data is already normalised
    TrX = double(X);
    TeX = double(XValid);
    N = size(TrX, 1);
    S = TrX * TrX' / N;
    [Vm, Dm] = eigs(S, M); % the M largest eigenvectors
    lm = Dm(1:size(Dm,1)+1:end); % or Dm * ones(M, 1)

    fprintf('[pcaManual] Computing eigenvalues for original huge matrix...\n');
    norm = sqrt(N * lm)';
    % Um = tmp ./ norm;
    tmp = TrX' * Vm;
    Um = zeros(size(tmp));
    for i = 1:size(tmp, 2)
        Um(:, i) = tmp(:, i) ./ norm(i);
    end

    TrZ = TrX * Um;
    TeZ = TeX * Um;
end

