function split = manualSplit(X, y)
    % Settup manual split
    K = 3;
    lim62 = 15.75;
    lim25 = 15.25;
        
    % Split indices
    % Indexes range over 1, 2 and 3
    idx62 = X(:, 25) >= lim62;
    idx25 = X(:, 62) < lim25;
    idx = idx62 + (idx25 & idx62) + 1;
    split.fullIdx = idx;
    
    % split the data
    for k = 1:K
        split.idxNo{k} = find(idx == k);
        split.X{k} = X(idx == k, :);
        split.y{k} = y(idx == k, :);
    end
end