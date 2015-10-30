function magellan()
    % A better explorator than Polo (hopefully)
    
    % SETTINGS
    splitSeed = 42;
    splitRatio = 0.7;
    clusterManuallyFlag = true;
    displayClustersFlag = true;
    displayResultsFlag = true;
    filterOutliersFlag = false;
%     method = @meanMethod;
%     method = @GDLSMethod;
    method = @linearRidgeKFoldMethod;
%     method = @emplifiedRidgeKFoldMethod;

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
        [XTr, yTr] = filterOutliers(XTr, yTr, idxTr);
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
    assert(false, 'not yet implemented');
end

function [XTr, yTr] = filterOutliers(XTr, yTr, idxTr)
    assert(false, 'not yet implemented');
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
        kXTr = XTr(idxTr == k);
        kyTr = yTr(idxTr == k);
        kXVa = XVa(idxVa == k);
        
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
    K = 10;
    if k == 1
        kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 1);
    elseif k == 2
        kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 2);
    else
        kyVaPred = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 2);
    end
end
