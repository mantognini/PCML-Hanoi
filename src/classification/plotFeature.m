function plotFeature(isCategorical, feature, X_train, X_test)
%PLOTFEATURE Plot the distribution of the given feature
%   If categorical is true then use categories to plot distribution
%
%   This function assumes a figure/subplot is open/selected.
%

    if (isCategorical)
        data_train = categorical(X_train(:, feature));
        data_test = categorical(X_test(:, feature));
    else
        data_train = X_train(:, feature);
        data_test = X_test(:, feature);
    end
    
    h_train = histogram(data_train);
    hold on;
    h_test = histogram(data_test);

    h_train.FaceColor = [1 0 0];
    h_test.FaceColor = [0 0 1];
    h_train.Normalization = 'probability';
    h_test.Normalization  = 'probability';

    xlabel('input categories');
    ylabel('probability');
    ylim([0 1]);

    legend('train', 'test', 'Location', 'north');

    title([num2str(feature) 'th feature']);
    grid on;
    %axis square;

    hold off;

end

