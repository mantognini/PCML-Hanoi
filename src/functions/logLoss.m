function error = logLoss(y, yPred)
% yPred is continous!
    assert(length(setdiff(unique(y), [0 1])) == 0);
    
    N = length(y);
    sig = sigmoid(yPred);
    error = -(y' * log(sig) + (1 - y)' * log(1 - sig)) / N;
end