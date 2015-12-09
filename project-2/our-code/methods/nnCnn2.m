function yPred = nnCnn2(train, XValid)
%
% Simply apply NN on HOG feature for binary discrimination

    % SETTINGS:
    INNER_SIZE = 10;
    EPOCHS     = 15;

    % Apply NN
    yPred = nn(INNER_SIZE, EPOCHS, 1, train.X.cnn, train.y, XValid.cnn);
    
end

