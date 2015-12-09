function [TrNormZ, TeNormZ] = pcaHog(M, train, XValid)
%
% Project data on the M more significant eigenvectors

    fprintf('[pcaHog] Extracting data & computing eigenvectors & forming data...\n');
    addpath(genpath('toolboxs/DeepLearnToolbox-master/')); % for zscore & normalize
    tic
    % ON NORMALISED DATA!
    [TrNormX, mu, sigma] = zscore(double(train.X.hog));
    TeNormX = normalize(double(XValid.hog), mu, sigma);
    S = cov(TrNormX, 1);
    % covariance matrix normalised by the number of samples N
    % the `1` above define the normalisation factor to 1/N
    
    [Um, ~] = eigs(S, M); % the M largest eigenvectors
    
    % convert X to subspace of size M
    TrNormZ = TrNormX * Um;
    TeNormZ = TeNormX * Um;
    toc

end

