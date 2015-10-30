function [XTr, yTr, XTe, yTe] = doSplit(y, X, prop)
% doSplit(y, X, prop)
%   Split the data according to prop.
%   Note that the data is not shuffled before the split. X and y are
%   splitted but kept in their original order.
%
    % Get indices
    N = size(y,1);
	idx = randperm(N);%1:N;
    Ntr = floor(prop * N);
    idxTr = idx(1:Ntr);
    idxTe = idx(Ntr+1:end);
    
	% Split the data
    XTr = X(idxTr,:);
    yTr = y(idxTr);
    XTe = X(idxTe,:);
    yTe = y(idxTe);
end
