function xPhied = emplifyFeatures051(X, ~)
    D = size(X, 2);
    assert(D == 67); % no features should have been removed
    
    % Special features
    discreteFeaturesIdx = [9 11 15 22 27 30 38 40 44 47 56 61];
    spottedFeaturesIdx = [53 16 4 59 43 20 14 18 46 33 1 5];
    continuousFeaturesIdx = setdiff(1:D, discreteFeaturesIdx);
    
    % Decide look of X
    degrees = [0.5 1];
    features{1} = continuousFeaturesIdx;
    features{2} = continuousFeaturesIdx;
    
    % Build X
    xPhied = [];
    for degIdx = 1:length(degrees)
        degree = degrees(degIdx);
        XSupplement = polynomialPhi(X(:, features{degIdx}), degree);
        xPhied = [xPhied XSupplement];
    end
end

