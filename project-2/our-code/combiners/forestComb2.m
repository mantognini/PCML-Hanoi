function yPred = forestComb2(X, y, XValid, ops)
%
% Predict using forests from piotr toolbox for binary y {0, 1}
    forest = forestTrain(X, y + 1, ops); % Piotr needs y = (1 or 2)
    [yPred, ~] = forestApply(single(XValid), forest);
    yPred = yPred - 1;
end