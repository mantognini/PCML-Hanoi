function sig = sigmoid(x)
%
% Sigmoidal function

    % Trick against numerical issue
    N = length(x);
    posIdx = find(x > 0);
    negIdx = setdiff(1:N, posIdx);
    
    xPos = x(posIdx);
    xNeg = x(negIdx);
    
    sig = zeros(N, 1);
    sig(posIdx) = ones(length(xPos), 1)  ./ (1 + exp(-xPos));
    sig(negIdx) = exp(xNeg) ./ (1 + exp(xNeg));
    
    % x(find(isnan(sig)))
end