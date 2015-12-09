function [TrZ, TeZ] = pcaHog(M, train, XValid)
%
% Project data on the M more significant eigenvectors

    fprintf('[pcaHog] Extracting data & computing eigenvectors & forming data...\n');
    tic
    % NOTE: data is already normalised
    S = cov(double(train.X.hog), 1);
    % covariance matrix normalised by the number of samples N
    % the `1` above define the normalisation factor to 1/N
    
    [Um, ~] = eigs(S, M); % the M largest eigenvectors
    
    % convert X to subspace of size M
    TrZ = train.X.hog * Um;
    TeZ = XValid.hog * Um;
    toc

end

