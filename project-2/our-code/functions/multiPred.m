function yPred = multiPred(X, y, XValid, iter, ratio, methods, combiner)
%
% Invoke the methods and combine their results to predict XValid.
% Methods are training {iter} times with {ratio}% of the data.
% The iter*{1-ratio}% of data produced is used for training the combiner.
% For the final prediction, the whole data (X, y) is used as training and
% it is the combiner which outputs the final prediction.

    % Generate samples for training the combiner
    XComb = [];
    yComb = [];
    for j = 1:iter
        % Split the data into (Tr, Te)
        [XTr, yTr, XTe, yTe] = splitHogCnn(X, y, ratio);

        % Train methods on (Tr) and predict (Te)
        m = length(methods);
        newXTr = zeros(length(yTe), m); % their output on (Te)
        for i = 1:m
            method = methods{i};

            % Call methods
            newXTr(:, i) = method(XTr, yTr, XTe);
        end
        
        % Save predictions
        XComb = [XComb; newXTr];
        yComb = [yComb; yTe];
    end

    % Train methods on (X, y) and predict (Valid)
    newXValid = zeros(size(XValid.hog, 1), m);
    for i = 1:m
        method = methods{i};

        % Call methods
        newXValid(:, i) = method(X, y, XValid); % their output on (Valid)
    end
    
    % Combine the results to predict XValid
    yPred = combiner(XComb, yComb, newXValid);

end