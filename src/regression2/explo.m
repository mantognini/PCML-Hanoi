
clear all;
polo = Polo();

[K, clusters] = polo.clusterize();

% Found by removing one feature after the other manually on 1+x+x^2+x^3
% model (x per feature) (37 features)
features{1} = [ 4  6  7  10  11  12  13  14  16  19  21  22  24  26  28  29  30  35  36  37  38  42  44  45  46  49  50  51  53  54  55  56  57  58  59  64  66 ];

% Best using 1+x+x^2+x^3 model
features{2} = setdiff(1:67, [... % all features minus the following:
    9 11 15 22 27 30 38 40 44 47 56 61 ... % discretes
    1 2 3 7 8 10 12 24 26 32 33 34 36 41 45 46 49 50 51 52 54 55 57 64 66 ... % "bad" features for lin+square
    5 6 16 21 25 28 42 53 58 62 ... % "bad" features for cube
]);
% 20 "important" feature for k = 2 ->
% 4 13 14 17 18 19 20 23 29 31 35 37 39 43 48 59 60 63 65 67

% Found by removing one feature after the other manually on 1+x+x^2+x^3
% model (x per feature) (26 features)
features{3} = [ 4 5 6 10 20 21 26 27 30 31 32 34 35 36 37 45 46 49 50 51 53 54 56 59 63 65 ];

methods = {
    %@polo.constantMethod,
    %@medianMethod,
    %@meanMethod,
    %@GDLSMethod,
    %@ridgeLinear10Fold,
    @polo.basisFunctionsMethod,
};

M = numel(methods);
S = 10;
splitRatio = 0.7;

k = 3;
features{k} = [1 1:67];
best = 100000;
for suspect = features{k}
    clear e rmse;
    oldFs = features{k};
    features{k} = setdiff(features{k}, suspect);
    
    fprintf(['processing suspect ' num2str(suspect) '\n']);

%fprintf('I am starting...\n');

%for k = 1%1:K
    %fprintf(['processing cluster ' num2str(k) ' of ' num2str(K) '\n']);
    
    cluster = clusters{k};
    cluster = polo.deleteYOutliers(cluster);
    
%     polo.plotData(['Cluster ' num2str(k)], cluster);
%     continue;
    
    cluster = polo.trimFeatures(cluster, features{k});
    
    % Test each methods several times with different training and 
    % validation split of the data
    fprintf('\tprocessing seed ');
    clear e rmse;
    for seed = 1:S
        fprintf([num2str(seed) '  ']);
        setSeed(seed);

        % Split data into training and validation sets
        N = size(cluster.train.X, 1);
        idx = randperm(N);
        X = cluster.train.X(idx, :);
        y = cluster.train.y(idx);

        [XTr, yTr, XValid, yValid] = doSplit(y, X, splitRatio);
        
        % Test each method
        for methodNo = 1:numel(methods)
            %fprintf(['\t\tprocessing method ' num2str(methodNo) ' of ' num2str(numel(methods)) '\n']);
            method = methods{methodNo};

            % Collect predictions
            yValidPred = method(XTr, yTr, XValid);

            % Compute error
            rmse(k, seed, methodNo) = computeRmse(yValidPred - yValid);
        end
    end
        
    % Plot RMSE for this cluster
    figure('Name', ['RMSE for cluster ' num2str(k)]);
    tmp = rmse(k, :, :);
    tmp = reshape(tmp, S, M);
    boxplot(tmp);
    names = cellfun(@func2str, methods, 'UniformOutput', false);
    legend(findobj(gca,'Tag','Box'), names);
    title([num2str(k) 'th cluster']);
    xlabel('methods');
    ylabel('RMSE');
%end

fprintf('\nI am nearly finished...\n');

% Combine each cluster error together
e = rmse .^ 2;
e = mean(e, 1);
e = e .^ 0.5;
e = reshape(e, S, M);

% figure('Name', 'Overall RMSE');
% boxplot(e, 1:M);
% names = cellfun(@func2str, methods, 'UniformOutput', false);
% legend(findobj(gca,'Tag','Box'), names);
% xlabel('methods');
% ylabel('RMSE');
% %ylim([30 40]);
% 
% fprintf('I am finished now!\n');

med = median(e);
fprintf(['suspect: ' num2str(suspect) ' - median: ' num2str(med) ' for (' ...
    num2str(length(features{k})) ') ' num2str(features{k}) '.\n']);

if (med < best)
    fprintf('\tbetter!\n');
    best = med;
elseif (med > best)
    fprintf('\tworse\n');
    features{k} = oldFs;
else
    fprintf('\tequal\n');
    features{k} = oldFs;
end

end % masta loop on suspect

display(features{k});

