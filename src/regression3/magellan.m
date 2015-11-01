function magellan(arg)
    % A better explorator than Polo (hopefully)
    
    if strcmp(arg, 'box')
        doBoxplot();
    elseif strcmp(arg, 'lambda')
        lambdaPlot();
    elseif strcmp(arg, 'learning')
        learningCurve();
    else
        fprintf('Unknown argument\n');
    end
end

function learningCurve()
    % Given the best learners for regression, compute rmse
    
    % initial parameters
    ratios = 0.6:0.025:0.9;
    seeds = 20;
    clusterManuallyFlag = false;
    
    % define method
%     method = @linearRidgeKFoldMethod;
%     method = @emplifiedRidgeKFoldMethod2;
    method = @finalMethod;
    
    % Compute
    validRMSE = zeros(length(seeds), length(ratios));
    trainRMSE = zeros(length(seeds), length(ratios));
    for r = 1:length(ratios)
        ratio = ratios(r);
        
        fprintf(['Ratio ' num2str(r) 'th ']);
        
        for s = 1:seeds
            fprintf('.');
            [rmseTr, rmseVa] = runMethod(method, clusterManuallyFlag, false, ratio);
            validRMSE(s, r) = rmseVa;
            trainRMSE(s, r) = rmseTr;
        end
        fprintf('\n');
    end
    
    figure('Name', 'Learning curve');
    rmse = reshape([validRMSE; trainRMSE], 2, seeds, length(ratios));
    aboxplot(rmse, 'labels', ratios, 'colorgrad', 'orange_down');
    xlabel('Training set size');
    ylabel('RMSE');
    legend('train', 'valid');
    title(['Learning curve for ' func2str(method)]);
end

function lambdaPlot()
    K = 10;
    degrees = 1:6;
    clusterManuallyFlag = false;
    clusters = 1:3;
    S = 20; % if 1 -> lambda curves, otherwise boxplot
    
    if S > 1
        lambdas = logspace(-5, 12, 20); % not too many point here
    else
        lambdas = logspace(-5, 12, 100);
    end
    
    for cluster = clusters
        kmseTe = zeros(length(degrees), S, length(lambdas));
        fprintf(['cluster ' num2str(cluster) ':\n']);
        for s = 1:S
            fprintf(['\tseed ' num2str(s, '%02.0f') ' ']);
            for d = degrees
                fprintf('.');
                [XTr, yTr, ~] = loadData();
                XVa = XTr; % just a trick for clusterize
                XTe = XTr; % just a trick for clusterize
                [idxTr, ~, ~] = clusterize(clusterManuallyFlag, XTr, yTr, XVa, XTe);
                clear XVa XTe;
                
                %removeCategorical(XTr);
                XTr(:, [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61]) = [];
                [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr);
                
                kXTr = XTr(idxTr == cluster, :);
                kyTr = yTr(idxTr == cluster, :);
                
                kN = length(kyTr);
                ktXTr = [ones(kN, 1) polynomialPhi(kXTr, d)];
                
                % Compute K-Fold CV indices
                idx = randperm(kN);
                Nk = floor(kN / K);
                D = size(ktXTr, 2) - 1;
                
                N_train = Nk * (K - 1);
                assert(N_train >= D, ['Problem N_train < D: N_train = ' num2str(N_train) ', D = ' num2str(D)]);
                
                clear idxCV;
                for k = 1:K
                    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
                end
                
                % K-fold cross validation
                kmseTeSub = zeros(1, K);
                for l = 1:length(lambdas)
                    kLambda = lambdas(l);
                    
                    for k = 1:K
                        % get k'th subgroup in test, others in train
                        idxTe = idxCV(k,:);
                        idxTr = idxCV([1:k-1 k+1:end],:);
                        idxTr = idxTr(:);
                        yTe = kyTr(idxTe);
                        tXTe = ktXTr(idxTe,:);
                        yTr = kyTr(idxTr);
                        tXTr = ktXTr(idxTr,:);
                        
                        % find beta & compute rmse
                        beta = ridgeRegression(yTr, tXTr, kLambda);
                        kmseTeSub(k) = computeRmse(yTe - tXTe * beta);
                    end
                    
                    kmseTe(d, s, l) = mean(kmseTeSub);
                end
            end % degree

            if S == 1
                figure('Name', ['cluster ' num2str(cluster) ': lambda/test rmse']);
                kmseTe = reshape(kmseTe, length(degrees), length(lambdas));
                semilogx(lambdas, kmseTe, '-', 'LineWidth', 4);
                xlabel('lambda');
                ylabel('Test RMSE');
                title(['cluster ' num2str(cluster)]);
                legend(arrayfun(@num2str, degrees'));
            end
        end % s
        fprintf('\n');

        if S > 1
            figure('Name', ['cluster ' num2str(cluster) ': lambda/test rmse']);
            colors = colormap(lines(length(degrees)));
            aboxplot(kmseTe, 'Colormap', colors);
            xlabel('lambda');
            ylabel('Test RMSE');
            title(['cluster ' num2str(cluster)]);
            legend(arrayfun(@num2str, degrees'), 'location', 'northwest');
            set(gca, 'XTickLabel', lambdas);
        end
    end
end

function doBoxplot()

    methods = {
        % method, manual cluster, remove outliers, name %
        { @overallMeanMethod, true, false, 'overall mean' },
        
        { @meanMethod, true, false, 'mean' },
        
        { @GDLSMethod, true, false, 'GDLS' },
        { @GDLSMethod, true, true, 'GDLS - outliers' },
        
        { @linearRidgeKFoldMethod, true, false, 'linear rigde' },
        { @linearRidgeKFoldMethod, true, true, 'linear rigde - outliers' },
        
%         { @emplifiedRidgeKFoldMethod1, true, false, 'emplified ridge 1' },
%         { @emplifiedRidgeKFoldMethod1, true, true, 'emplified ridge 1 - outliers' },
%         { @emplifiedRidgeKFoldMethod2, true, false, 'emplified ridge 2' },
%         { @emplifiedRidgeKFoldMethod2, true, true, 'emplified ridge 2 - outliers' },
        { @emplifiedRidgeKFoldMethod3, true, false, 'emplified ridge 3' },
        { @emplifiedRidgeKFoldMethod3, true, true, 'emplified ridge 3 - outliers' },
%         { @emplifiedRidgeKFoldMethod4, true, false, 'emplified ridge 4' },
%         { @emplifiedRidgeKFoldMethod4, true, true, 'emplified ridge 4 - outliers' },

        { @finalMethod, true, false, 'phis' },
        { @finalMethod, true, true, 'phis - outliers' },
        
        { @GDLSMethod, false, false, 'GDLS + auto' },
        { @linearRidgeKFoldMethod, false, false, 'linear rigde + auto' },
%         { @emplifiedRidgeKFoldMethod1, false, false, 'emplified ridge 1 + auto' },
%         { @emplifiedRidgeKFoldMethod2, false, false, 'emplified ridge 2 + auto' },
        { @emplifiedRidgeKFoldMethod3, false, false, 'emplified ridge 3 + auto' },
%         { @emplifiedRidgeKFoldMethod4, false, false, 'emplified ridge 4 + auto' },
        { @finalMethod, false, false, 'phis + auto' },
    };

    M = numel(methods);
    S = 50;
    splitRatio = 0.7;
    
    rmse = zeros(S, M);
    for s = 1:S
        fprintf(['seed ' num2str(s, '%02.0f') ' ']);
        for m = 1:M
            fprintf('.');
            [rmse(s, m), ~] = runMethod(methods{m}{1}, methods{m}{2}, methods{m}{3}, splitRatio);
        end
        fprintf('\n');
    end
    
    figure;
    boxplot(rmse, 1:M);
    methodNames = cellfun(@(x) x{4}, methods, 'UniformOutput', false);
    for i = 1:M
        methodNames{i} = [num2str(i) ' ' methodNames{i}];
    end
    legend(findobj(gca,'Tag','Box'), methodNames);
    xlabel('methods');
    ylabel('RMSE');
end

function [rmseTr, rmseVa] = runMethod(method, clusterManuallyFlag, filterOutliersFlag, splitRatio)
    % SETTINGS
%     displayClustersFlag = true;
%     displayResultsFlag = true;
    displayClustersFlag = false;
    displayResultsFlag = false;

    [X_train, y_train, ~] = loadData();
    
    % Split into training & validation sets
    %setSeed(splitSeed);
    [XTr, yTr, XVa, yVa] = doSplit(y_train, X_train, splitRatio);
    
    % Clusterize data
    XTe = XVa; % just a trick for clusterize
    [idxTr, idxVa, ~] = clusterize(clusterManuallyFlag, XTr, yTr, XVa, XTe);
    
    % Display clusterized data
    if (displayClustersFlag)
        figure('Name', ['clustering manual? ' bool2str(clusterManuallyFlag)]);
        subplot(1, 2, 1);
        displayData(XTr, yTr, 25, 62, idxTr, 'training');
        subplot(1, 2, 2);
        displayData(XVa, yVa, 25, 62, idxVa, 'validation');
        
        figure('Name', 'Histogram of response');
        displayYHistogram(yTr, idxTr, 'training');
    end
    
    % Remove (or not) outliers
    if (filterOutliersFlag)
        [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr);
    end

    % Collect predictions
    [yTrPred, yVaPred] = applyMethodOnClusters(method, XTr, yTr, XVa, idxTr, idxVa);
    rmseTr = computeRmse(yTrPred - yTr);
    rmseVa = computeRmse(yVaPred - yVa);
%     for k = 1:3
%         krmse(k) = computeRmse(yVaPred(idxVa == k) - yVa(idxVa == k));
%     end
%     fprintf(['method ' func2str(method) ' -> RMSE = ' num2str(rmse) '\n']);
%     fprintf(['RMSE per cluster: ' num2str(krmse) '\n']);

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

function displayYHistogram(y, idx, name)
    for k = 1:3
        histogram(y(idx == k), 50);
        hold on;
    end
    xlabel('response');
    ylabel('count');
    title(name);
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

function [XTr, yTr, idxTr] = filterOutliers(XTr, yTr, idxTr)

    % Remove Y-outliers
    STD = 2; % keep 95%
    for k = 1:3
        kSigma = std(yTr(idxTr == k));
        kMu = mean(yTr(idxTr == k));

        idx = abs(yTr(idxTr == k) - kMu) >= STD * kSigma;

%         dels = length(find(idx));
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

function [yTrPred, yVaPred] = applyMethodOnClusters(method, XTr, yTr, XVa, idxTr, idxVa)
    yTrPred = zeros(size(XTr, 1), 1);
    yVaPred = zeros(size(XVa, 1), 1);
    
    for k = 1:3
        kXTr = XTr(idxTr == k, :);
        kyTr = yTr(idxTr == k, :);
        kXVa = XVa(idxVa == k, :);
        
        [kyTrPred, kyVaPred] = method(kXTr, kyTr, kXVa, k, XTr, yTr);
        
        yTrPred(idxTr == k) = kyTrPred;
        yVaPred(idxVa == k) = kyVaPred;
    end
end

function [kyTrPred, kyVaPred] = overallMeanMethod(kXTr, ~, kXVa, ~, ~, yTr)
    overallMean = mean(yTr);
    kyTrPred = ones(size(kXTr, 1), 1) * overallMean;
    kyVaPred = ones(size(kXVa, 1), 1) * overallMean;
end

function [kyTrPred, kyVaPred] = meanMethod(kXTr, kyTr, kXVa, ~, ~, ~)
    clusterMean = mean(kyTr);
    kyTrPred = ones(size(kXTr, 1), 1) * clusterMean;
    kyVaPred = ones(size(kXVa, 1), 1) * clusterMean;
end

function [kyTrPred, kyVaPred] = GDLSMethod(kXTr, kyTr, kXVa, ~, ~, ~)
    kNTr = size(kXTr, 1);
    kNVa = size(kXVa, 1);

    [kXTr, kXVa] = normalizeBoth(kXTr, kXVa);

    ktXTr = [ones(kNTr, 1) kXTr];
    kBeta = leastSquaresGDLS(kyTr, ktXTr);
    kyTrPred = ktXTr * kBeta;

    ktXVa = [ones(kNVa, 1) kXVa];
    kyVaPred = ktXVa * kBeta;
end

function [kyTrPred, kyVaPred] = linearRidgeKFoldMethod(kXTr, kyTr, kXVa, ~, ~, ~)
    K = 10;
    [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 1);
end

function [kyTrPred, kyVaPred] = emplifiedRidgeKFoldMethod1(kXTr, kyTr, kXVa, k, ~, ~)
    K = 10;
    if k == 1
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 2
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 3
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 1);
    end
end

function [kyTrPred, kyVaPred] = emplifiedRidgeKFoldMethod2(kXTr, kyTr, kXVa, k, ~, ~)
    K = 10;
    if k == 1
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 2
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 3
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 2);
    end
end

function [kyTrPred, kyVaPred] = emplifiedRidgeKFoldMethod3(kXTr, kyTr, kXVa, k, ~, ~)
    K = 10;
    if k == 1
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 2
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 3
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    end
end

function [kyTrPred, kyVaPred] = emplifiedRidgeKFoldMethod4(kXTr, kyTr, kXVa, k, ~, ~)
    K = 10;
    if k == 1
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 2
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 3);
    elseif k == 3
        [kyTrPred, kyVaPred] = predictRidgeKFold(kXTr, kyTr, kXVa, K, @polynomialPhi, 4);
    end
end

function [kyTrPred, kyVaPred] = finalMethod(kXTr, kyTr, kXVa, k, ~, ~)
    kXTe = kXVa; % just a trick for finalMethod_impl
    [kyTrPred, kyVaPred, ~] = finalMethod_impl(kXTr, kyTr, kXVa, kXTe, k);
end

