function [combiStar, errStar, errors] = crossValid(X, y, K, combi, modelFn, errFn)    
%
% K Cross-validation on different combinations of parameters using
% a model and an error function
% Each line of combi is a parameter combination to K-Fold
    % Check K
    assert(K > 1);

    % Define parts of the cake
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N / K);
    idxCV = zeros(K, Nk);
    for k = 1:K
        idxCV(k, :) = idx(1 + (k - 1) * Nk : k * Nk);
    end
    
    % for each parameter combination
    nbCombi = size(combi, 1);
    errors = zeros(nbCombi, 1);
    parfor n = 1:nbCombi
        params = combi(n, :);
        error = zeros(K, 1);
        
        % K-Fold
        for k = 1:K
            fprintf(['param ' num2str(n) '/' num2str(nbCombi) ', fold ' num2str(k) '/' num2str(K) '\n']);
            
            % compute indices
            idxTe = idxCV(k, :);
            idxTr = idxCV([1:k-1 k+1:end], :);
            idxTr = idxTr(:);
            
            % compute data
            XTr = X(idxTr, :);
            yTr = y(idxTr);
            XTe = X(idxTe, :);
            yTe = y(idxTe);

            % compute model
            yPred = modelFn(XTr, yTr, XTe, params);
            
            % compute error
            error(k) = errFn(yTe, yPred);
        end
        
        % Average errors
        errors(n) = mean(error);
    end
    
    % Find best combination
    [errStar, iStar] = min(errors);
    combiStar = combi(iStar, :);
end
