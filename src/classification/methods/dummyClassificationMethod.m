function yValidPred = dummyClassificationMethod(XTr, yTr, XValid)
%DUMMYCLASSIFICATIONMETHOD always classify the data with the same class
%

    N = size(XValid, 1);
    yValidPred = ones(N, 1); % class "1"

end

