function yPred = randM4(train, XValid)
    NValid = size(XValid.hog, 1);
    
    % random predictions
    yValues = unique(train.y);
    yPred = yValues(floor(rand(NValid, 1) * length(yValues) + 1));
end