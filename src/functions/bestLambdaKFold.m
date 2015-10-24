function lambda = bestLambdaKFold(y, tX, K)
% bestLambdaKFold(y,tX)
%   Compute the best lambda of ridge regression using K-Fold on each lambda
%   and taking the minimum avg error estimate.
%
    % Define lambdas range
    lambdas = logspace(-4, 4, 50);

    % Compute K-Fold CV indices
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N / K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    % K-fold cross validation
    mseTr = zeros(1, length(lambdas));
    mseTe = zeros(1, length(lambdas));
    mseTrSub = zeros(1, K);
    mseTeSub = zeros(1, K);
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

            % find beta & compute rmse
            beta = ridgeRegression(yTr, tXTr, lambda);
            mseTrSub(k) = computeRmse(yTr - tXTr * beta);
            mseTeSub(k) = computeRmse(yTe - tXTe * beta);
        end

        mseTr(l) = mean(mseTrSub);
        mseTe(l) = mean(mseTeSub);
    end
    
    % Best lambda
    [minMeanTe, lambdaIdStars] = min(mseTe(:));
    lambda = lambdas(lambdaIdStars);
end

