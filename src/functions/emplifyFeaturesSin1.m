function xPhied = emplifyFeaturesSin1(X, ~)
    D = size(X, 2);
    
    % Decide look of X
    xPhied = [];
    XSupplement = sin(X);
    xPhied = [xPhied XSupplement];
    XSupplement = X;
    xPhied = [xPhied XSupplement];
end

