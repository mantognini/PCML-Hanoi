function yPred = svmPca(X, y, XValid, M, kernelF, C, params)
%
% svm on pca'd X, XValid with fixed M, C and params
    
    % pca
    [X2, XValid2] = pcaManual(X, XValid, M);
    
    % svm
    yPred = svmF(X2, y, XValid2, kernelF, C, params);
    
end
