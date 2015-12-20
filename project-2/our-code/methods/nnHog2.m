function yPred = nnHog2(train, XValid, category)
%
% Simply apply NN on HOG feature for binary discrimination

    % Make y binary
    train.y = toBinary(train.y, category);

    % SETTINGS:
    INNER_SIZE = 10;
    EPOCHS     = 15;

    % Apply NN
    yPred = nn(INNER_SIZE, 2, EPOCHS, 1, double(train.X.hog), train.y, XValid.hog);
    
end

