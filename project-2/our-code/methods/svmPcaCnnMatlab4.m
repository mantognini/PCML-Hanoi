function yPred = svmPcaCnnMatlab4(train, XValid)
%
%

    M = 150;
    [TrZ, VaZ] = pcaManual(train.X.cnn, XValid.cnn, M);

    svmDefault = templateSVM('Verbose', 0);
    model = fitcecoc(TrZ, train.y, 'Learners', svmDefault);

    [yPred, ~] = predict(model, VaZ);

end

