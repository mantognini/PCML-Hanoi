function yValidPred = medianMethod(XTr, yTr, XValid)
% medianMethod(XTr, yTr, XValid)
%   Predict the median of training outputs.
%
    medianY = median(yTr);
    N = size(XValid, 1);
    yValidPred = ones(N, 1)*medianY;
end
