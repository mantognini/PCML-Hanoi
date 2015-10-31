function error = zeroOneLoss(y, yPred)
% yPred is discrete 0,1!
    assert(length(setdiff(unique(yPred), [0 1])) == 0);
    assert(length(setdiff(unique(y), [0 1])) == 0);
    
    N = length(yPred);
    error = sum(mod((yPred + y), 2)) / N;
end