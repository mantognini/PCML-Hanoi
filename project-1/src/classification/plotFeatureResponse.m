function plotFeatureResponse(feature, X_train, y_train)
%PLOTFEATURERESPONSE Plot the distribution on the given features with
%respect to the response
%
%   This function assumes a figure/subplot is open/selected.
%

    data_train = X_train(:, feature);
    
    h = histogram2(data_train, y_train);
    %h.Normalization = 'probability';
    
    legend('train', 'Location', 'northeast');
    
    xlabel('input');
    ylabel('response');
    zlabel('count');
    %zlabel('probability');
    %zlim([0 1]);
    
    title([num2str(feature) 'th feature']);

end

