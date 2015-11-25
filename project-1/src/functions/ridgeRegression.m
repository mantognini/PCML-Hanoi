function beta = ridgeRegression(y,tX, lambda)
% ridgeRegression(y,tX, lambda)
%   Compute the ridge regression with the given lambda.
%
    % Matrix multiplying lambda
    % Note: we are not penalizing beta0
    tI = eye(size(tX, 2));
    tI(1, 1) = 0;
    
    % Compute best beta
    beta = (tX' * tX + lambda * tI) \ tX' * y;
end

