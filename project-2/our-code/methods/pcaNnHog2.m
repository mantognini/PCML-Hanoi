function yPred = pcaNnHog2(train, XValid)
%
% Apply PCA followed by NN on HOG feature

    % SETTINGS:
    M          = 200; % might be lower maybe
    INNER_SIZE = 10;
    EPOCHS     = 15;
    
    % Apply PCA
    [TrNormZ, TeNormZ] = pcaHog(M, train, XValid);

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 1, TrNormZ, train.y, TeNormZ);
end

