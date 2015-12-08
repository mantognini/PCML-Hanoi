function yPred = randM4(train, XValid)
%
% Predict uniformly {1, 2, 3, 4}
    NValid = size(XValid.hog, 1);
    
    % random predictions
    yValues = unique(train.y);
    yPred = yValues(floor(rand(NValid, 1) * length(yValues) + 1));
end