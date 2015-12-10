function yPred = pcaNnHog4(train, XValid)
%
% Apply PCA followed by NN on HOG feature

    % SETTINGS:
    M             = 200; % might be lower maybe
    INNER_SIZE    = 10;
    EPOCHS        = 15;
    LEARNING_RATE = 2;   % might be non-optimal
    
    % Apply PCA
    [TrZ, TeZ] = pcaHog(M, train, XValid);

    % Apply NN
    yPred = nn(INNER_SIZE, LEARNING_RATE, EPOCHS, 0, TrZ, train.y, TeZ);
end

