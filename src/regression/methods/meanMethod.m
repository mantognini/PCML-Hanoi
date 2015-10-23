function yValidPred = meanMethod(XTr, yTr, XValid)
% meanMethod(XTr, yTr, XValid)
%   Predict the mean of training outputs.
%
    meanY = mean(yTr);
    N = size(XValid, 1);
    yValidPred = ones(N, 1)*meanY;
end
