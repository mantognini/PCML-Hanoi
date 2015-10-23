function split = manualSplit(X)
% manualSplit(X, y)
%   Split the data into three clusters manually defined with lines.
%
    % Settup manual split
    K = 3;
    lim62 = 15.75;
    lim25 = 15.25;
        
    % Split indices
    % Indexes range over 1, 2 and 3
    idx62 = X(:, 25) >= lim62;
    idx25 = X(:, 62) < lim25;
    idx = idx62 + (idx25 & idx62) + 1;
    
    % split the data
    for k = 1:K
        split.idx{k} = find(idx == k);
    end
end
