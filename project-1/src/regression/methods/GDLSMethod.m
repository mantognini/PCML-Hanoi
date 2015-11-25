function yValidPred = GDLSMethod(XTr, yTr, XValid)
% GDLSMethod(XTr, yTr, XValid)
%   Predict output by using the gradient descent method on the training
%   data and line search to find alpha.
%

    % Compute gradient descent
    D = size(XTr, 2);
    
    X = normalize(XTr);
    tX = [ones(size(X, 1), 1) X];
    y = yTr;
    beta = leastSquaresGDLS(y, tX);

    % Predict outputs for validation set
    X = normalize(XValid);
    tX = [ones(size(X, 1), 1) X];
    yValidPred = tX * beta;
end

