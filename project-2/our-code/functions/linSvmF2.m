function yPred = linSvmF2(X, y, XValid, C)
%
% Linear svm using a fixed C
    yPred = svmF2(X, y, XValid, @linearKernel, C, []);
end