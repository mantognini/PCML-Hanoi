function yPred = svmF2(X, y, XValid, kernelFn, C, params)
%
% Binary decision on svmF function
    yPred = svmF(X, y, XValid, kernelFn, C, params);
    
    % Binary decision
    otherIdx = (yPred < 0);
    yPred(otherIdx) = 0;
    yPred(~otherIdx) = 1;
end