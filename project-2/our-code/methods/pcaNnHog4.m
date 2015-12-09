function yPred = pcaNnHog4(train, XValid)
%
% Apply PCA followed by NN on HOG feature

    % SETTINGS:
    M          = 200; % might be lower maybe
    INNER_SIZE = 10;
    EPOCHS     = 15;

    % Extract M eigenvectors
    fprintf('[PcaNnHog4] Extracting data & computing eigenvectors & forming data...\n');
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

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 0, TrNormZ, train.y, TeNormZ);
end

