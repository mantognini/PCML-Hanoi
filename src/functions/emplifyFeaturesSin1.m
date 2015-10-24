function xPhied = emplifyFeaturesSin1(X, ~)
    D = size(X, 2);
    assert(D == 67); % no features should have been removed
    
    % Special features
    discreteFeaturesIdx = [9 11 15 22 27 30 38 40 44 47 56 61];
    spottedFeaturesIdx = [53 16 4 59 43 20 14 18 46 33 1 5];
    continuousFeaturesIdx = setdiff(1:D, discreteFeaturesIdx);
    
    % Decide look of X
    xPhied = [];
    XSupplement = sin(X);
    xPhied = [xPhied XSupplement];
    XSupplement = X;
    xPhied = [xPhied X];
end

