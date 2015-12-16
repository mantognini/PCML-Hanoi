function yPred = multiPred(X, y, XValid, ratio, methods, combiner)
%
% Invoke the methods and combine their results to predict XValid

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
    
    % Train methods on (X, y) and predict (Valid)
    for i = 1:m
        method = methods{i};
        
        % Call methods
        newXValid(:, i) = method(X, y, XValid); % their output on (Valid)
    end
    
    % Combine the results to predict XValid
    yPred = combiner(newXTr, yTe, newXValid);

end