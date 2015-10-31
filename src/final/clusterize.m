
function [idxTr, idxVa, idxTe] = clusterize(clusterManuallyFlag, XTr, yTr, XVa, XTe)
    if (clusterManuallyFlag)
        [idxTr, idxVa, idxTe] = manualClustering(XTr, yTr, XVa, XTe);
    else
        [idxTr, idxVa, idxTe] = autoClustering(XTr, yTr, XVa, XTe);
    end
end

function [idx] = manualClustering_impl(X)
    % Apply manual splitting of the identified three input sources
    % Use feature 25 and 62 for that.
    lim62 = 15.75;
    lim25 = 15.25;

    X25 = X(:, 25);
    X62 = X(:, 62);

    idx62 = X62 >= lim62;
    idx25 = X25 < lim25;

    % Indexes range over 1, 2 and 3
    idx = idx62 + (idx25 & idx62) + 1;
end

function [idxTr, idxVa, idxTe] = manualClustering(XTr, ~, XVa, XTe)
   idxTr = manualClustering_impl(XTr);
   idxVa = manualClustering_impl(XVa);
   idxTe = manualClustering_impl(XTe);
end

function [idxTr, idxVa, idxTe] = autoClustering(XTr, yTr, XVa, XTe)
    XTr = [ XTr(:, 25) XTr(:, 62) ];
    XVa = [ XVa(:, 25) XVa(:, 62) ];
    XTe = [ XTe(:, 25) XTe(:, 62) ];

    % Clusterize data
    K = 3;
    C = [ 12, 12, 1800 ; 12, 18, 5000 ; 17, 18, 8000 ];
    idxTr = kmeans([XTr yTr], K, 'MaxIter', 1000, 'Start', C);

    % Compute mean and variance
    mus = zeros(K, 2);
    sigmas = zeros(K, 2);
    for k = 1:K
        kMu25 = mean(XTr(idxTr == k, 1));
        kMu62 = mean(XTr(idxTr == k, 2));

        kStd25 = std(XTr(idxTr == k, 1));
        kStd62 = std(XTr(idxTr == k, 2));

        mus(k, :) = [ kMu25 , kMu62 ];
        sigmas(k, :) = [ kStd25 , kStd62 ];
    end

    % Compute probabilities of being in a cluster
    pVa = zeros(size(XVa, 1), K);
    pTe = zeros(size(XVa, 1), K);
    for k = 1:K
        pVa(:, k) = mvnpdf(XVa, mus(k, :), sigmas(k, :));
        pTe(:, k) = mvnpdf(XTe, mus(k, :), sigmas(k, :));
    end

    [~, idxVa] = max(pVa, [], 2);
    [~, idxTe] = max(pTe, [], 2);
end


