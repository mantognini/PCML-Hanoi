classdef FinalMethod
    % Use the apply function to build prediction for validation and test data
    % (basis functions)
    
    properties
        features % {1,2,3}: list of feature to keep
        debug % logical value, if on display graphs
        clusterManual % if true -> manual otherwise kmeans based
    end
    
    methods
        function obj = FinalMethod(debug, clusterManual)
            obj.features{1} = [ 4 6 7 10 11 12 13 14 16 19 21 22 24 26 28 29 ...
                30 35 36 37 38 42 44 45 46 49 50 51 53 54 55 56 57 58 59 ...
                64 66 ];
            obj.features{2} = [ 4 13 14 17 18 19 20 23 29 31 35 37 39 43 ...
                48 59 60 63 65 67];
            obj.features{3} = [ 1 3 5 7 9 15 26 30 31 32 33 34 38 40 42 44 ...
                46 50 56 57 58 61 ];
            
            obj.debug = debug;
            obj.clusterManual = clusterManual;
        end
        
        % Apply best overall strategy
        function [yValid, yTest] = apply(self, XTrain, yTrain, XValid, XTest)
            % Create empty vector for output prediction
            yValid = zeros(size(XValid, 1), 1);
            yTest = zeros(size(XTest, 1), 1);
            
            % Indexes range over 1, 2 and 3 for our three clusters
            if self.clusterManual
                idx_train = self.clusterIndex(XTrain);
                idx_valid = self.clusterIndex(XValid);
                idx_test = self.clusterIndex(XTest);
            else
                [idx_train, idx_valid, idx_test] = self.clusterIndex2(XTrain, yTrain, XValid, XTest);
            end

            % Split training, validation and testing data & group them into
            % the three clusters we manually identified
            K = 3;
            if (self.debug)
            subplot(2, 2, 4);
            end
            for k = 1:K
%                 fprintf(['k = ' num2str(k) '\n']);
                
                cluster.train.X = XTrain(idx_train == k, :);
                cluster.train.y = yTrain(idx_train == k, :);
                
                cluster.valid.X = XValid(idx_valid == k, :);
                
                cluster.test.X = XTest(idx_test == k, :);
            
                % Remove misclassified training data
                [i, cluster] = self.deleteYOutliers(cluster);
%                 i = 0;
                [j, cluster] = self.deleteXOutliers(cluster);
%                 j = 0;
                fprintf(['removed ' num2str(i + j) ' outliers for cluster ' num2str(k) '\n']);
                
                if (self.debug)
                plot3(cluster.train.X(:, 25), ... % X25, training, cluster k
                      cluster.train.X(:, 62), ... % X62, training, cluster k
                      cluster.train.y,        ... % y,   training, cluster k
                      '.', 'MarkerSize', 30);
                hold on;
                end
                
                % Remove features that we believe are problematic
                %cluster = self.trimFeatures(cluster, self.features{k});
                
%                 % Normalise training, validation and test using training's 
%                 % mean and variance
%                 mu = mean(cluster.train.X);
% 
%                 muV = repmat(mu, size(cluster.train.X, 1), 1);
%                 cluster.train.X = cluster.train.X - muV;
% 
%                 muV = repmat(mu, size(cluster.valid.X, 1), 1);
%                 cluster.valid.X = cluster.valid.X - muV;
%                 
%                 muV = repmat(mu, size(cluster.test.X, 1), 1);
%                 cluster.test.X = cluster.test.X - muV;
% 
%                 sigma = std(cluster.train.X);
%                 if sigma ~= 0
%                     sigmaV = repmat(sigma, size(cluster.train.X, 1), 1);
%                     cluster.train.X = cluster.train.X ./ sigmaV;
% 
%                     sigmaV = repmat(sigma, size(cluster.valid.X, 1), 1);
%                     cluster.valid.X = cluster.valid.X ./ sigmaV;
% 
%                     sigmaV = repmat(sigma, size(cluster.test.X, 1), 1);
%                     cluster.test.X = cluster.test.X ./ sigmaV;
%                 end
                
                % Build basis functions
                D = size(cluster.train.X, 2);
                phis = self.buildPhis(D, k);
                
                % Apply basis functions
                cluster.train.tX = self.map(phis, cluster.train.X);
                cluster.valid.tX = self.map(phis, cluster.valid.X);
                cluster.test.tX = self.map(phis, cluster.test.X);
                
                % Find model parameters
                cluster.lambda = bestLambdaKFold(cluster.train.y, cluster.train.tX, 10);
%                 display(cluster.lambda, 'lambda');
                cluster.beta = ridgeRegression(cluster.train.y, cluster.train.tX, cluster.lambda);
%                 display(cluster.beta, 'beta');
                
                % Compute prediction
                cluster.valid.y = cluster.valid.tX * cluster.beta;
                cluster.test.y = cluster.test.tX * cluster.beta;
                
                % Combine prediction
                yValid(idx_valid == k) = cluster.valid.y;
                yTest(idx_test == k) = cluster.test.y;
            end
            
            if (self.debug)
            xlabel('25th feature');
            xlim([10 20]);
            ylabel('62th feature');
            ylim([0 30]);
            zlabel('response');
            title('TRAINING - outliers');
            grid on;
            end
        end
        
        function [idxTrain, idxValid, idxTest] = clusterIndex2(self, XTrain, yTrain, XValid, XTest)
            XTr = [ XTrain(:, 25) XTrain(:, 62) ];
            XVa = [ XValid(:, 25) XValid(:, 62) ];
            XTe = [ XTest(:, 25) XTest(:, 62) ];
            
            % Clusterize data
            K = 3;
            C = [ 12, 12, 1800 ; 12, 18, 5000 ; 17, 18, 8000 ];
            idxTrain = kmeans([XTr yTrain], K, 'MaxIter', 1000, 'Start', C);
%             idxTrain = cluster(fitgmdist([XTr yTrain], K), [XTr yTrain]); % this variante can achieve better results but can be unstable

            % Print result
            if (self.debug)
            figure('Name', 'Clustering using response');
            subplot(2, 2, 1);
            end
            mus = zeros(K, 2);
            sigmas = zeros(K, 2);
            for k = 1:K
                if (self.debug)
                plot3(XTr(idxTrain == k, 1), ... % X25, training, cluster k
                      XTr(idxTrain == k, 2), ... % X62, training, cluster k
                      yTrain(idxTrain == k), ... % y,   training, cluster k
                      '.', 'MarkerSize', 30);
                hold on;
                end

                mu25 = mean(XTr(idxTrain == k, 1));
                mu62 = mean(XTr(idxTrain == k, 2));

                std25 = std(XTr(idxTrain == k, 1));
                std62 = std(XTr(idxTrain == k, 2));

                mus(k, :) = [ mu25 , mu62 ];
                sigmas(k, :) = [ std25 , std62 ];
            end
            if (self.debug)
            xlabel('25th feature');
            xlim([10 20]);
            ylabel('62th feature');
            ylim([0 30]);
            zlabel('response');
            title('TRAINING');
            grid on;
            % axis square;
            end

            % Compute probabilites of being in a cluster
            pVa = zeros(size(XVa, 1), K);
            pTe = zeros(size(XTe, 1), K);
            for k = 1:K
                pVa(:, k) = mvnpdf(XVa, mus(k, :), sigmas(k, :));
                pTe(:, k) = mvnpdf(XTe, mus(k, :), sigmas(k, :));
            end

            [~, idxValid] = max(pVa, [], 2);
            [~, idxTest] = max(pTe, [], 2);
            
            
            if (self.debug)
            subplot(2, 2, 2);
            for k = 1:K
                plot(XVa(idxValid == k, 1), ... % X25, validation, cluster k
                     XVa(idxValid == k, 2), ... % X62, validation, cluster k
                     '.', 'MarkerSize', 30);
                hold on;
            end
            xlabel('25th feature');
            xlim([10 20]);
            ylabel('62th feature');
            ylim([0 30]);
            title('VALIDATION');
            grid on;
            % axis square;
            
            subplot(2, 2, 3);
            for k = 1:K
                plot(XTe(idxTest == k, 1), ... % X25, validation, cluster k
                     XTe(idxTest == k, 2), ... % X62, validation, cluster k
                     '.', 'MarkerSize', 30);
                hold on;
            end
            xlabel('25th feature');
            xlim([10 20]);
            ylabel('62th feature');
            ylim([0 30]);
            title('TEST');
            grid on;
            % axis square;
            end
        end
        
        function idx = clusterIndex(~, X)
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
        
        function [dels, data] = deleteXOutliers(~, data)
            STD = 4;
            
            %X = data.train.X;
            X = [ data.train.X(25, :) data.train.X(62, :) ];
            
            sigma = std(X);
            mu = mean(X);
            
            muV = repmat(mu, size(X, 1), 1);
            sigmaV = repmat(sigma, size(X, 1), 1);
            
            idx = abs(X - muV) >= STD * sigmaV;
            idx = any(idx, 2);
            
            dels = length(find(idx));

            data.train.X(idx, :) = [];
            data.train.y(idx, :) = [];
        end
        
        function [dels, data] = deleteYOutliers(~, data)
            STD = 3;
            
            y = data.train.y;
            
            sigma = std(y);
            mu = mean(y);
            
            idx = abs(y - mu) >= STD * sigma;
            
            dels = length(find(idx));

            data.train.X(idx, :) = [];
            data.train.y(idx, :) = [];
        end
        
        function data = trimFeatures(~, data, keepIdx)
            D = size(data.train.X, 2);
            
            removeIdx = setdiff(1:D, keepIdx);
%             display(removeIdx, 'removeIdx');
            
            data.train.X(:, removeIdx) = [];
            data.valid.X(:, removeIdx) = [];
            data.test.X(:, removeIdx) = [];
        end
        
        function phis = buildPhis(~, D, k)
            % Build basis functions
            power = @(i, x) x .^ i;
            
            p = 1;
            phis{p} = @(x) 1;
            
            for d = 1:D
                if k == 1
%                     p = p + 1; phis{p} = @(x) sign(x(d)) * power(1/2, sign(x(d)) * x(d));
                    p = p + 1; phis{p} = @(x) power(1, x(d));
                    p = p + 1; phis{p} = @(x) power(3, x(d));
                    p = p + 1; phis{p} = @(x) power(5, x(d));
                end
                
                if k == 2
%                     p = p + 1; phis{p} = @(x) sign(x(d)) * power(1/2, sign(x(d)) * x(d));
%                     p = p + 1; phis{p} = @(x) power(1, x(d));
                    p = p + 1; phis{p} = @(x) power(3, x(d));
                end
                
                if k == 3
%                     p = p + 1; phis{p} = @(x) sign(x(d)) * power(1/2, sign(x(d)) * x(d));
%                     p = p + 1; phis{p} = @(x) power(1, x(d));
                    p = p + 1; phis{p} = @(x) power(3, x(d));
                end
            end
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
    end
end

