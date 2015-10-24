function xPhied = emplifyFeatures12(X, ~)
    D = size(X, 2);
    
    % Special features
    allIdx = 1:D;
    discreteFeaturesIdx = getDiscreteFeaturesIdx(X, 5);
    
    % Decide look of X
    degrees = [1 2];
    features{1} = allIdx;
    features{2} = allIdx(~discreteFeaturesIdx);
    
    % Build X
    xPhied = [];
    for degIdx = 1:length(degrees)
        degree = degrees(degIdx);
        XSupplement = polynomialPhi(X(:, features{degIdx}), degree);
        xPhied = [xPhied XSupplement];
    end
end

