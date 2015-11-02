
function [yVaPred, yTePred, rmseTr] = finalMethodRegression(XTr, yTr, XVa, XTe)
    manually = true;
    [idxTr, idxVa, idxTe] = clusterize(manually, XTr, yTr, XVa, XTe);
    
    [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr);
    
    yTrPred = zeros(size(XTr, 1), 1);
    yVaPred = zeros(size(XVa, 1), 1);
    yTePred = zeros(size(XTe, 1), 1);
    
    for k = 1:3
        kXTr = XTr(idxTr == k, :);
        kyTr = yTr(idxTr == k, :);
        kXVa = XVa(idxVa == k, :);
        kXTe = XTe(idxTe == k, :);
        
        [kyTrPred, kyVaPred, kyTePred] = finalMethodRegression_impl(kXTr, kyTr, kXVa, kXTe, k);
        
        yTrPred(idxTr == k) = kyTrPred;
        yVaPred(idxVa == k) = kyVaPred;
        yTePred(idxTe == k) = kyTePred;
    end

    rmseTr = computeRmse(yTrPred - yTr);
end

function [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr)

    % Remove Y-outliers
    STD = 3; % keep 97.5%
    for k = 1:3
        kSigma = std(yTr(idxTr == k));
        kMu = mean(yTr(idxTr == k));

        idx = abs(yTr(idxTr == k) - kMu) >= STD * kSigma;

        XTr(idx, :) = [];
        yTr(idx, :) = [];
        idxTr(idx) = [];
    end
end

