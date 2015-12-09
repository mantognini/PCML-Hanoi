function yPred = PcaNnHog4(train, XValid)
%
% Apply PCA followed by NN on HOG feature
% TODO: this should be split into two functions:
%  - one for PCA on HOG, and
%  - one for NN (with parameters for HOG or CNN)

    % SETTINGS:
    M           = 200; % might be lower maybe
    INPUT_SIZE  = M;
    INNER_SIZE  = 10;
    OUTPUT_SIZE = 4;
    EPOCHS      = 15;

    % FIRST FUNCTION: EXTRACT EIGENVECTORS
    fprintf('[PcaNnHog4] Extracting data & computing eigenvectors & forming data...\n');
    addpath(genpath('toolboxs/DeepLearnToolbox-master/'));
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

    % SECOND FUNCTION: THE NEURAL NETWORK
    fprintf('[PcaNnHog4] Training NN & predicting classes...\n');
    tnn = tic;
    
    % Setup NN.
    nn = nnsetup([INPUT_SIZE INNER_SIZE OUTPUT_SIZE]);

    opts.numepochs  = EPOCHS;
    opts.batchsize  = 100;
    opts.plot       = 0;      % silent training
    nn.learningRate = 2;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor(size(TrNormZ) / opts.batchsize);
    TrNormZ      = TrNormZ(1:numSampToUse, :);
    labels       = train.y(1:numSampToUse);

    % prepare labels for NN
    LL = [1 * (labels == 1), ... % first column, p(y=1)
          1 * (labels == 2), ... % second column, p(y=2), etc
          1 * (labels == 3), ...
          1 * (labels == 4) ];

    [nn, ~] = nntrain(nn, TrNormZ, LL, opts);

    % to get the scores we need to do nnff (feed-forward)
    nn.testing = 1;
    nn = nnff(nn, TeNormZ, zeros(size(TeNormZ, 1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPredZ = nn.a{end};
    [~, yPred] = max(nnPredZ, [], 2); % get the most probable class

    toc(tnn)
end

