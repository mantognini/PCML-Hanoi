function yPred = pcaNnCnn2(train, XValid)
%
% Apply PCA followed by NN on CNN feature

    % SETTINGS:
    M          = 200; % might be non-optimal
    INNER_SIZE = 100;
    EPOCHS     = 30;

    % Apply PCA
    [TrZ, TeZ] = pcaCnn(M, train, XValid);

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 1, TrZ, train.y, TeZ);
end

