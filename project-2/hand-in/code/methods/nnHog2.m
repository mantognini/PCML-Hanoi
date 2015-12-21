function yPred = nnHog2(train, XValid, category)
%
% Simply apply NN on HOG feature for binary discrimination

    if category ~= 4
        error('unsupported');
    end

    % SETTINGS:
    INNER_SIZE = 10;
    EPOCHS     = 20;

    % Apply NN
    yPred = nn(INNER_SIZE, 2, EPOCHS, 1, double(train.X.hog), train.y, XValid.hog);
    
end

