
function [yVaPred, yTePred, rmseTr] = finalMethod(XTr, yTr, XVa, XTe)
    manually = true;
    [idxTr, idxVa, idxTe] = clusterize(manually, XTr, yTr, XVa, XTe);
    
    yTrPred = zeros(size(XTr, 1), 1);
    yVaPred = zeros(size(XVa, 1), 1);
    yTePred = zeros(size(XTe, 1), 1);
    
    for k = 1:3
        kXTr = XTr(idxTr == k, :);
        kyTr = yTr(idxTr == k, :);
        kXVa = XVa(idxVa == k, :);
        kXTe = XTe(idxTe == k, :);
        
        [kyTrPred, kyVaPred, kyTePred] = finalMethod_impl(kXTr, kyTr, kXVa, kXTe, k);
        
        yTrPred(idxTr == k) = kyTrPred;
        yVaPred(idxVa == k) = kyVaPred;
        yTePred(idxTe == k) = kyTePred;
    end

    rmseTr = computeRmse(yTrPred - yTr);
end

