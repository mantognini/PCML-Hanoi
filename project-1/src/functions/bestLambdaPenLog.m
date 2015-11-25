function lambda = bestLambdaPenLog(y, tX, K)
%
    % Define lambdas range
    lambdas = logspace(-2, 4, 10);

    % Compute K-Fold CV indices
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N / K);
    D = size(tX, 2) - 1;
    
    N_train = Nk * (K - 1);
    assert(N_train >= D, ['Problem N_train < D: N_train = ' num2str(N_train) ', D = ' num2str(D)]);
    
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    % K-fold cross validation
    mseTe = zeros(1, length(lambdas));
    mseTr = zeros(1, length(lambdas));
    mseTeSub = zeros(1, K);
    mseTrSub = zeros(1, K);
    for l = 1:length(lambdas)
        lambda = lambdas(l);

        for k = 1:K
            % get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe = y(idxTe);
            tXTe = tX(idxTe,:);
            yTr = y(idxTr);
            tXTr = tX(idxTr,:);

            % beta & predict
            beta = PLRLS(yTr, tXTr, lambda);
            mseTeSub(k) = logLoss(yTe, tXTe * beta);
            mseTrSub(k) = logLoss(yTr, tXTr * beta);
        end

        mseTe(l) = mean(mseTeSub);
        mseTr(l) = mean(mseTrSub);
    end
    
    % Best lambda
    [~, lambdaIdStars] = min(mseTe(:));
    lambda = lambdas(lambdaIdStars);
    
    % Plot
%     figure;
%     semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');

end
