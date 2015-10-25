
clear all;
polo = Polo();

[K, clusters] = polo.clusterize();
features{1} = [ 53 16 4 ];
features{2} = [ 59 43 20 18 14 ];
features{3} = [ 46 5 1 ];

methods = {
    @polo.constantMethod,
    @medianMethod,
    @meanMethod,
    @GDLSMethod,
    @ridgeLinear10Fold,
    @polo.basisFunctionsMethod,
};

M = numel(methods);
S = 10;
splitRatio = 0.7;

fprintf('I am starting...\n');

for k = 1:K
    fprintf(['processing cluster ' num2str(k) ' of ' num2str(K) '\n']);
    
    cluster = clusters{k};
    cluster = polo.deleteYOutliers(cluster);
    
%     polo.plotData(['Cluster ' num2str(k)], cluster);
%     continue;
    
    cluster = polo.trimFeatures(cluster, features{k});
    
    % Test each methods several times with different training and 
    % validation split of the data
    for seed = 1:S
        fprintf(['\tprocessing seed ' num2str(seed) ' of ' num2str(S) '\n']);
        setSeed(seed);

        % Split data into training and validation sets
        N = size(cluster.train.X, 1);
        idx = randperm(N);
        X = cluster.train.X(idx, :);
        y = cluster.train.y(idx);

        [XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);
        
        % Test each method
        for methodNo = 1:numel(methods)
            fprintf(['\t\tprocessing method ' num2str(methodNo) ' of ' num2str(numel(methods)) '\n']);
            method = methods{methodNo};

            % Collect predictions
            yValidPred = method(XTr, yTr, XValid);

            % Compute error
            rmse(k, seed, methodNo) = computeRmse(yValidPred - yValid);
        end
    end
        
%     % Plot RMSE for this cluster
%     figure('Name', ['RMSE for cluster ' num2str(k)]);
%     tmp = rmse(k, :, :);
%     tmp = reshape(tmp, S, M);
%     boxplot(tmp);
%     names = cellfun(@func2str, methods, 'UniformOutput', false);
%     legend(findobj(gca,'Tag','Box'), names);
%     title([num2str(k) 'th cluster']);
%     xlabel('methods');
%     ylabel('RMSE');
end

fprintf('I am nearly finished...\n');

% Combine each cluster error together
e = rmse .^ 2;
e = mean(e, 1);
e = e .^ 0.5;
e = reshape(e, S, M);

figure('Name', 'Overall RMSE');
boxplot(e, 1:M);
names = cellfun(@func2str, methods, 'UniformOutput', false);
legend(findobj(gca,'Tag','Box'), names);
xlabel('methods');
ylabel('RMSE');

fprintf('I am finished now!\n');

    
   

