function yPred = knnComb(X, y, XValid, k)
%
% Predict using knn
    knnModel = fitcknn(X, y, 'NumNeighbors', k, 'Standardize', 1);
    yPred = predict(knnModel, XValid);
end