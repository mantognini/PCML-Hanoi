function K = rbfKernel(X1, X2, gamma)
%
% Radial Basis Function

    % Efficient way
     K = exp(-gamma .* pdist2(X1, X2));
end

