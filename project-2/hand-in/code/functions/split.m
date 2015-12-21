function [XTr, yTr, XTe, yTe] = split(X, y, ratio)
%
% Split (X, y) randomly into training and testing
    N = size(y, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTr = idx(1:splitIdx);
    idxTe = idx(splitIdx + 1:end);
    
    XTr = X(idxTr, :);
    yTr = y(idxTr);
    XTe = X(idxTe, :);
    yTe = y(idxTe);
end