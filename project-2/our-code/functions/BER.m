function error = BER(y, yPred)
% Balanced Error Rate (BER)
    labels = unique(y);
    nbLabels = length(labels);
    errors = zeros(nbLabels, 1);
    for i = 1:nbLabels
        label = labels(i);
        labelIndices = find(y == label);
        nbLabelIndices = length(labelIndices);
        errors(i) = sum(y(labelIndices) ~= yPred(labelIndices)) / nbLabelIndices;
    end
    error = sum(errors) / nbLabels;
end