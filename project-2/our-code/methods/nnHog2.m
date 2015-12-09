function yPred = nnHog2(train, XValid)
%
% Simply apply NN on HOG feature for binary discrimination

    % SETTINGS:
    INNER_SIZE = 10;
    EPOCHS     = 15;

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 1, train.X.hog, train.y, XValid.hog);
    
end

