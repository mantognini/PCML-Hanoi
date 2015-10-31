function magellan()
    % A better explorator than Polo (hopefully)

    methods = {
        % method, manual cluster, remove outliers, name %
        { @meanMethod, true, false, 'mean' },
        { @GDLSMethod, true, false, 'GDLS' },
%         { @GDLSMethod, true, true, 'GDLS - outliers' },
%         { @GDLSMethod, false, false, 'GDLS - auto' },
%         { @GDLSMethod, false, true, 'GDLS - auto - outliers' },
%         { @linearRidgeKFoldMethod, true, false, 'linear rigde' },
%         { @linearRidgeKFoldMethod, true, true, 'linear rigde - outliers' },
%         { @linearRidgeKFoldMethod, false, false, 'linear rigde - auto' },
%         { @linearRidgeKFoldMethod, false, true, 'linear rigde - auto - outliers' },
%         { @emplifiedRidgeKFoldMethod, true, false, 'emplified ridge' },
%         { @emplifiedRidgeKFoldMethod, true, true, 'emplified ridge - outliers' },
%         { @emplifiedRidgeKFoldMethod, false, false, 'emplified ridge - auto' },
%         { @emplifiedRidgeKFoldMethod, false, true, 'emplified ridge - auto -outliers' },
%         { @finalMethod, true, false, 'phis' },
%         { @finalMethod, true, true, 'phis - outliers' },
%         { @finalMethod, false, false, 'phis - auto' },
%         { @finalMethod, false, true, 'phis - auto - outliers' },
    };

    M = numel(methods);
    S = 1;
    global rmse; % keep it alive between runs
    rmse = zeros(S, M);
    for s = 1:S
        for m = 1:M
            rmse(s, m) = runMethod(methods{m}{1}, methods{m}{2}, methods{m}{3});
        end
    end
    
    figure;
    boxplot(rmse, 1:M);
    methodNames = cellfun(@(x) x{4}, methods, 'UniformOutput', false);
    legend(findobj(gca,'Tag','Box'), methodNames);
    xlabel('methods');
    ylabel('RMSE');
end


function [rmse] = runMethod(method, clusterManuallyFlag, filterOutliersFlag)
    % SETTINGS
    %splitSeed = 42;
    splitRatio = 0.7;
    displayClustersFlag = false;%true;
    displayResultsFlag = false;%true;

    [X_train, y_train, ~] = loadData();
    
    % Split into training & validation sets
    %setSeed(splitSeed);
    [XTr, yTr, XVa, yVa] = doSplit(y_train, X_train, splitRatio);
    
    % Clusterize data
    if (clusterManuallyFlag)
        [idxTr, idxVa] = manualClustering(XTr, yTr, XVa);
    else
        [idxTr, idxVa] = autoClustering(XTr, yTr, XVa);
    end
    
    % Display clusterized data
    if (displayClustersFlag)
        figure('Name', ['clustering manual? ' bool2str(clusterManuallyFlag)]);
        subplot(1, 2, 1);
        displayData(XTr, yTr, 25, 62, idxTr, 'training');
        subplot(1, 2, 2);
        displayData(XVa, yVa, 25, 62, idxVa, 'validation');
    end
    
    % Remove (or not) outliers
    if (filterOutliersFlag)
        [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr);
    end

    % Collect predictions
    yVaPred = applyMethodOnClusters(method, XTr, yTr, XVa, idxTr, idxVa);
    rmse = computeRmse(yVaPred - yVa);
    for k = 1:3
        krmse(k) = computeRmse(yVaPred(idxVa == k) - yVa(idxVa == k));
    end
    fprintf(['method ' func2str(method) ' -> RMSE = ' num2str(rmse) '\n']);
    fprintf(['RMSE per cluster: ' num2str(krmse) '\n']);

    if (displayResultsFlag)
        figure('Name', 'Prediction on validation set');
        psVa = displayData(XVa, yVa, 25, 62, idxVa, '');
        psVaPred = displayData(XVa, yVaPred, 25, 62, idxVa, '');
        legend([psVa psVaPred], ...
               'data cluster 1', 'data cluster 2', 'data cluster 3', ...
               'pred cluster 1', 'pred cluster 2', 'pred cluster 3', ...
               'location', 'northeast');
    end
end

function [str] = bool2str(b)
    if b
        str = 'true';
    else
        str = 'false';
    end
end

function [X_train, y_train, X_test] = loadData()
    load('HaNoi_regression.mat');
end

function [ps] = displayData(X, y, f1, f2, idx, name)
    % Assume figure/subplot setup
    for k = 1:3
        ps(k) = plot3(X(idx == k, f1), X(idx == k, f2), y(idx == k), '.', 'MarkerSize', 30);
        hold on;
    end
    xlabel([num2str(f1) 'th feature']);
    ylabel([num2str(f2) 'th feature']);
    zlabel('response');
    zlim([0 15000]);
    title(name);
    grid on;
%     axis square;
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

function [idxTr, idxVa] = manualClustering(XTr, yTr, XVa)
   idxTr = manualClustering_impl(XTr);
   idxVa = manualClustering_impl(XVa);
end

function [idxTr, idxVa] = autoClustering(XTr, yTr, XVa)
    XTr = [ XTr(:, 25) XTr(:, 62) ];
    XVa = [ XVa(:, 25) XVa(:, 62) ];

    % Clusterize data
    K = 3;
    C = [ 12, 12, 1800 ; 12, 18, 5000 ; 17, 18, 8000 ];
    idxTr = kmeans([XTr yTr], K, 'MaxIter', 1000, 'Start', C);

    % Print result
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

    % Compute probabilites of being in a cluster
    global pVa;
    pVa = zeros(size(XVa, 1), K);
    for k = 1:K
        pVa(:, k) = mvnpdf(XVa, mus(k, :), sigmas(k, :));
    end

    [~, idxVa] = max(pVa, [], 2);
end

function [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr)

    % Remove Y-outliers
    STD = 2; % keep 95%
    for k = 1:3
        kSigma = std(yTr(idxTr == k));
        kMu = mean(yTr(idxTr == k));

        idx = abs(yTr(idxTr == k) - kMu) >= STD * kSigma;

        dels = length(find(idx));
%         disp(dels);

        XTr(idx, :) = [];
        yTr(idx, :) = [];
        idxTr(idx) = [];
    end
end

function [XTr, XVa] = normalizeBoth(XTr, XVa)
    % Normalise XTr & XVa using XTr mean and variance
    mu = mean(XTr);

    muV = repmat(mu, size(XTr, 1), 1);
    XTr = XTr - muV;

    muV = repmat(mu, size(XVa, 1), 1);
    XVa = XVa - muV;

    sigma = std(XTr);
    if sigma ~= 0
        sigmaV = repmat(sigma, size(XTr, 1), 1);
        XTr = XTr ./ sigmaV;

        sigmaV = repmat(sigma, size(XVa, 1), 1);
        XVa = XVa ./ sigmaV;
    end
end

function [yVaPred] = applyMethodOnClusters(method, XTr, yTr, XVa, idxTr, idxVa)
    yVaPred = zeros(size(XVa, 1), 1);
    
    for k = 1:3
        kXTr = XTr(idxTr == k, :);
        kyTr = yTr(idxTr == k, :);
        kXVa = XVa(idxVa == k, :);
        
        yVaPred(idxVa == k) = method(kXTr, kyTr, kXVa, k);
    end
end

function [kyVaPred] = meanMethod(~, kyTr, kXVa, ~)
    clusterMean = mean(kyTr);
    kyVaPred = ones(size(kXVa, 1), 1) * clusterMean;
end

function [kyVaPred] = GDLSMethod(kXTr, kyTr, kXVa, ~)
    kNTr = size(kXTr, 1);
    kNVa = size(kXVa, 1);

    [kXTr, kXVa] = normalizeBoth(kXTr, kXVa);

    ktXTr = [ones(kNTr, 1) kXTr];
    kBeta = leastSquaresGDLS(kyTr, ktXTr);

    ktXVa = [ones(kNVa, 1) kXVa];
    kyVaPred = ktXVa * kBeta;
end

function [kyVaPred] = linearRidgeKFoldMethod(kXTr, kyTr, kXVa, ~)
    K = 10;
    kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 1);
end

function [kyVaPred] = emplifiedRidgeKFoldMethod(kXTr, kyTr, kXVa, k)
    K = 5;
    if k == 1
        kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 2
        kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 4);
    elseif k == 3
        kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    end
end

function [kyVaPred] = finalMethod(kXTr, kyTr, kXVa, k)
    fm = FinalMethod(false, true);
    D = size(kXTr, 2);
    kPhis = fm.buildPhis(D, k);
    ktXTr = fm.map(kPhis, kXTr);
    ktXVa = fm.map(kPhis, kXVa);
    
    K = 10;
    kLambda = bestLambdaKFold(kyTr, ktXTr, K);
    kBeta = ridgeRegression(kyTr, ktXTr, kLambda);
    
    kyVaPred = ktXVa * kBeta;
end
