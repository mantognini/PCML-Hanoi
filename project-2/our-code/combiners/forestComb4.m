function yPred = forestComb4(X, y, XValid, ops)
%
% Predict using forests from piotr toolbox for y {1, 2, 3, 4}
    forest = forestTrain(X, y, ops);
    [yPred, ~] = forestApply(single(XValid), forest);
end