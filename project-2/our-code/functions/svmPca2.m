function yPred = svmPca2(X, y, XValid, M, kernelF, C, params)
%
% binary decision on svmPca function
    
    % svmPca
    yPred = svmPca(X, y, XValid, M, kernelF, C, params);
    
    % Binary decision
    otherIdx = (yPred < 0);
    yPred(otherIdx) = 0;
    yPred(~otherIdx) = 1;
    
end
