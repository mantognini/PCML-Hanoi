function K = rbfKernel(X1, X2, gamma)
%
% Radial Basis Function

    % Efficient way
    K = exp(-gamma .* pdist2(double(X1), double(X2)));
    
    % Warning: Depending on your X1, X2, gamma, K might be close to I
    % In this case, you have to better tune gamma
end

