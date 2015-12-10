function yPred = nn(innerSize, learningRate, epochs, isBinary, TrX, Try, TeX)
%
% Apply NN on data
%
% Note
%   Try is always multiclass but yPred can be binary when requested

    fprintf('[nn] Training NN & predicting classes...\n');
    addpath(genpath('toolboxs/DeepLearnToolbox-master/'));
    tnn = tic;
    
    inputSize = size(TrX, 2);
    if isBinary
        outputSize = 2;
    else
        outputSize = 4;
    end
    
    % Setup NN.
    nn = nnsetup([inputSize innerSize outputSize]);

    opts.numepochs  = epochs;
    opts.batchsize  = 100;
    opts.plot       = 0;      % silent training
    nn.learningRate = learningRate;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor(size(TrX) / opts.batchsize);
    TrX          = TrX(1:numSampToUse, :);
    Try          = Try(1:numSampToUse);

    % prepare labels for NN
    if isBinary
        LL = [ 1*(Try == 4), ... % class 4 only
               1*(Try ~= 4) ];   % either class 1, 2 or 3
    else
        LL = [ 1*(Try == 1), ... % first column, p(y=1)
               1*(Try == 2), ... % second column, p(y=2), etc
               1*(Try == 3), ...
               1*(Try == 4) ];
    end

    [nn, ~] = nntrain(nn, TrX, LL, opts);

    % to get the scores we need to do nnff (feed-forward)
    nn.testing = 1;
    nn = nnff(nn, TeX, zeros(size(TeX, 1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPred = nn.a{end};
    [~, yPred] = max(nnPred, [], 2); % get the most probable class
    
    if isBinary
        yPred = yPred - 1; % map {1, 2} to {0, 1}
    end

    toc(tnn)
end

