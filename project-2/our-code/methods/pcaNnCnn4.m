function yPred = pcaNnCnn4(train, XValid)
%
% Apply PCA followed by NN on CNN feature

    % SETTINGS:
    M          = 200; % might be non-optimal
    INNER_SIZE = 100;
    EPOCHS     = 30;

    % Apply PCA
    [TrNormZ, TeNormZ] = pcaCnn(M, train, XValid);

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 0, TrNormZ, train.y, TeNormZ);
end

