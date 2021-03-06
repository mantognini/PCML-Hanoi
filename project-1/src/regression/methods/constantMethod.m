function yValidPred = constantMethod(XTr, yTr, XValid)
% constantMethod(XTr, yTr, XValid)
%   Predict the same constant for each data point.
%
    c = 4000; % by visual inspection
    N = size(XValid, 1);
    yValidPred = ones(N, 1)*c;
end
