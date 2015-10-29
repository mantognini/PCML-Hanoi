classdef FinalMethod
    % Use the apply function to build prediction for validation and test data
    % (basis functions)
    
    properties
        features % {1,2,3}: list of feature to keep
    end
    
    methods
        function obj = FinalMethod()
            obj.features{1} = [ 4 6 7 10 11 12 13 14 16 19 21 22 24 26 28 29 ...
                30 35 36 37 38 42 44 45 46 49 50 51 53 54 55 56 57 58 59 ...
                64 66 ];
            obj.features{2} = [ 4 13 14 17 18 19 20 23 29 31 35 37 39 43 ...
                48 59 60 63 65 67];
            obj.features{3} = [ 1 3 5 7 9 15 26 30 31 32 33 34 38 40 42 44 ...
                46 50 56 57 58 61 ];
        end
        
        % Apply best overall strategy
        function [yValid, yTest] = apply(self, XTrain, yTrain, XValid, XTest)
            % Create empty vector for output prediction
            yValid = zeros(size(XValid, 1), 1);
            yTest = zeros(size(XTest, 1), 1);
            
            % Indexes range over 1, 2 and 3 for our three clusters
            idx_train = self.clusterIndex(XTrain);
            idx_valid = self.clusterIndex(XValid);
            idx_test = self.clusterIndex(XTest);
            
            D = size(XTrain, 2);
            phis = self.buildPhis(D);

            % Split training, validation and testing data & group them into
            % the three clusters we manually identified
            K = 3;
            for k = 1:K
                cluster.train.X = XTrain(idx_train == k, :);
                cluster.train.y = yTrain(idx_train == k, :);
                
                cluster.valid.X = XValid(idx_valid == k, :);
                
                cluster.test.X = XTest(idx_test == k, :);
            
                % Remove misclassified training data
                [~, cluster] = self.deleteYOutliers(cluster);
                
                % Remove features that we believe are problematic
                cluster = self.trimFeatures(cluster, self.features{k});
                
                % Apply basis functions
                cluster.train.tX = self.map(phis, cluster.train.X);
                cluster.valid.tX = self.map(phis, cluster.valid.X);
                cluster.test.tX = self.map(phis, cluster.test.X);
                
                % Find model parameters
                cluster.lambda = bestLambdaKFold(cluster.train.y, cluster.train.tX, 10);
                cluster.beta = ridgeRegression(cluster.train.y, cluster.train.tX, cluster.lambda);
                
                % Compute prediction
                cluster.valid.y = cluster.valid.tX * cluster.beta;
                cluster.test.y = cluster.test.tX * cluster.beta;
                
                % Combine prediction
                yValid(idx_valid == k) = cluster.valid.y;
                yTest(idx_test == k) = cluster.test.y;
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
        
        function [dels, data] = deleteYOutliers(~, data)
            y = data.train.y;
            sigma = std(y);
            mu = mean(y);
            idx = abs(y - mu) >= 2 * sigma;
            
            dels = length(find(idx));

            data.train.X(idx, :) = [];
            data.train.y(idx, :) = [];
        end
        
        function data = trimFeatures(~, data, keepIdx)
            D = size(data.train.X, 2);
            
            removeIdx = setdiff(1:D, keepIdx);
            
            data.train.X(:, removeIdx) = [];
            data.valid.X(:, removeIdx) = [];
            data.test.X(:, removeIdx) = [];
        end
        
        function phis = buildPhis(~, D)
            % Build basis functions
            power = @(i, x) x .^ i;
            
            p = 1;
            phis{p} = @(x) 1;
            
            for d = 1:D
                p = p + 1; phis{p} = @(x) power(1, x(d));
                p = p + 1; phis{p} = @(x) power(2, x(d));
                p = p + 1; phis{p} = @(x) power(3, x(d));
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

