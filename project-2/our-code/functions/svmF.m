function yPred = svmF(X, y, XValid, kernelFn, C, params)
%
% svm using a fixed kernel and parameters
    % Check params if any
    nbParams = size(params, 1);
    if nbParams > 0
        assert(size(params, 2) == 1);
        assert(nbParams == 1); % Limit to 1 parameter
    end
    
    % Normalize data
    X = normalize(X);
    XValid = normalize(XValid);
    
    % Binary train y {-1, 1}
    otherIdx = (y == 0);
    y(otherIdx) = -1;
    y(~otherIdx) = 1;
    
    % Find alpha star
    if nbParams > 0
        KTrain = kernelFn(X, X, params(1));
    else
        KTrain = kernelFn(X, X);
    end
    [alpha, beta0] = SMO(KTrain, y, C);
    indexSV = (alpha > 0);
    
    % Predict
    if nbParams > 0
        KValid = kernelFn(XValid, X(indexSV, :), params(1));
    else
        KValid = kernelFn(XValid, X(indexSV, :));
    end
    yPred = KValid * (alpha(indexSV) .* y(indexSV)) + beta0;
end
