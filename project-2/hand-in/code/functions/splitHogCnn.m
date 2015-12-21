function [XTr, yTr, XTe, yTe] = splitHogCnn(X, y, ratio)
%
% Split w.r.t the {hog, cnn} structure
    % Define indices
    N = size(y, 1);
    splitIdx = floor(N * ratio);

    idx = randperm(N);
    idxTr = idx(1:splitIdx);
    idxTe = idx(splitIdx + 1:end);
    
    % Split
    XTr.hog = X.hog(idxTr, :);
    XTr.cnn = X.cnn(idxTr, :);
    yTr = y(idxTr);
    XTe.hog = X.hog(idxTe, :);
    XTe.cnn = X.cnn(idxTe, :);
    yTe = y(idxTe);
end