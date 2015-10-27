classdef Polo
    % Polo is a well know explorator, and now he's up to a new challenge:
    % find good regression models for PCML.
    
    properties
        data % train.X, train.y and test.X
    end
    
    methods
        % Constructor; load the data
        function polo = Polo()
            load('HaNoi_regression.mat');

            data.train.X = X_train;
            data.train.y = y_train;
            data.test.X  = X_test;
            
            polo.data = data;
        end
        
        function yValid = constantMethod(polo, ~, ~, XValid)
            overallMean = mean(polo.data.train.y);
            yValid = ones(size(XValid, 1), 1) * overallMean;
        end
        
        function yValid = basisFunctionsMethod(polo, XTr, yTr, XValid)
            % Normalise XTr & XValid using XTr mean and variance
            mu = mean(XTr);
            
            muV = repmat(mu, size(XTr, 1), 1);
            XTr = XTr - muV;
            
            muV = repmat(mu, size(XValid, 1), 1);
            XValid = XValid - muV;
            
            sigma = std(XTr);
            if sigma ~= 0
                sigmaV = repmat(sigma, size(XTr, 1), 1);
                XTr = XTr ./ sigmaV;
                
                sigmaV = repmat(sigma, size(XValid, 1), 1);
                XValid = XValid ./ sigmaV;
            end
            
            % Build basis functions
            power = @(i, x) x .^ i;
            sigma = @(x) 1 / (1 + exp(-x));
            p = 1;
            phis{p} = @(x) 1;
            
            D = size(XTr, 2);
            for d = 1:D
                p = p + 1; phis{p} = @(x) power(1, x(d));
                %p = p + 1; phis{p} = @(x) power(0.5, x(d)); % might imply complex numbers
                p = p + 1; phis{p} = @(x) power(2, x(d));
                p = p + 1; phis{p} = @(x) power(3, x(d));
                %p = p + 1; phis{p} = @(x) power(4, x(d));
                %p = p + 1; phis{p} = @(x) tanh(x(d));
                %p = p + 1; phis{p} = @(x) sigma(x(d));
            end
        
            tXTrPhi = polo.map(phis, XTr);
            %beta = leastSquaresGDLS(yTr, tXTrPhi); % not good with unnormilized values
            lambda = bestLambdaKFold(yTr, tXTrPhi, 10);
            beta = ridgeRegression(yTr, tXTrPhi, lambda);
            
            tXValidPhi = polo.map(phis, XValid);
            yValid = tXValidPhi * beta;
        end
        
        
        function XPhi = map(~, phis, X)
            N = size(X, 1);
            M = numel(phis);
            
            XPhi = zeros(N, M);
            for n = 1:N
                for m = 1:M
                    x = X(n, :);
                    XPhi(n, m) = phis{m}(x);
                end
            end
        end
        
        
        % Each returned cluster has train.X, train.y and test.X fields
        function [K, clusters] = clusterize(polo)
            K = 3;
            
            % Apply manual splitting of the identified three input sources
            X25_train = polo.data.train.X(:, 25);
            X62_train = polo.data.train.X(:, 62);
            X25_test  = polo.data.test.X(:, 25);
            X62_test  = polo.data.test.X(:, 62);

            lim62 = 15.75;
            lim25 = 15.25;

            idx62_train = X62_train >= lim62;
            idx62_test  = X62_test  >= lim62;

            idx25_train = X25_train < lim25;
            idx25_test  = X25_test  < lim25;

            % Indexes range over 1, 2 and 3
            idx_train = idx62_train + (idx25_train & idx62_train) + 1;
            idx_test  = idx62_test  + (idx25_test  & idx62_test)  + 1;

            % Split training data & group them into clusters
            for k = 1:K
                clusters{k}.train.X = polo.data.train.X(idx_train == k, :);
                clusters{k}.test.X  = polo.data.test.X(idx_test == k, :);

                clusters{k}.train.y = polo.data.train.y(idx_train == k, :);
            end
        end
        
        
        function data = deleteYOutliers(~, data)
            y = data.train.y;
            sigma = std(y);
            mu = mean(y);
            idx = abs(y - mu) >= 2 * sigma;
            
            fprintf(['Removed ' num2str(length(find(idx))) ' outliers.\n']);

            data.train.X(idx, :) = [];
            data.train.y(idx, :) = [];
        end
        
        
        function data = trimFeatures(~, data, keepIdx)
            D = size(data.train.X, 2);
            removeIdx = setdiff(1:D, keepIdx);
            data.train.X(:, removeIdx) = [];
            data.test.X(:, removeIdx) = [];
        end
        
        
        function plotData(~, figTitle, data)
            D = size(data.train.X, 2);
            plotDim = [2, 2];
            plotPerFig = plotDim(1) * plotDim(2);
            nbFig = ceil(D / plotPerFig);

            assert(D < nbFig * plotPerFig);

            for figNo = 0:(nbFig - 1)
                figure('Name', [figTitle ' - ' num2str(figNo + 1) ' of ' num2str(nbFig)]);

                for subplotNo = 1:plotPerFig
                    plotNo = figNo * plotPerFig + subplotNo;

                    if plotNo <= D
                        subplot(plotDim(1), plotDim(2), subplotNo);

                        plot(data.train.X(:, plotNo), data.train.y, '.');
                        title(['feature #' num2str(plotNo)]);

%                         histogram2(data.train.X(:, plotNo), data.train.y, 'DisplayStyle','tile');
%                         xlabel(['feature #' num2str(plotNo)]);
%                         ylabel('response');
                        
                        grid on;
                        axis square;
                    end
                end
            end
        end
        
        
    end
    
end

