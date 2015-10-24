function [X, y] = outliersFilter(X, y)
%NOFILTER Do not filter the data at all
%

    D = size(X, 2);

    % Filter out outliers deduced by features analysis
    isInit = 0;
    for f = 1:D % For each feature
        
        feature = X(:, f);
        
        uniques = length(unique(feature));
        % Assuming there's at most 5 categories
        isDiscrete = uniques <= 5;
        
        if ~isDiscrete
            sigma = std(feature);
            mu = mean(feature);
            idx = abs(feature - mu) >= 4 * sigma; % 99.7%

            % Combine outliers indices
            if isInit
                idx_outliers = idx_outliers | idx;
            else
                idx_outliers = idx;
            end
            isInit = 1;
        end
    end
    
    % Filter out outliers deduced by output analysis
    sigma = std(y);
    mu = mean(y);
    idx = abs(y - mu) >= 4 * sigma; % 99.7%

    % Combine outliers indices
    idx_outliers = idx_outliers | idx;
    
    fprintf(['Removing ' num2str(length(find(idx_outliers))) ' outliers...\n']);

    % Remove outliers
    X(idx_outliers, :) = [];
    y(idx_outliers, :) = [];

end

