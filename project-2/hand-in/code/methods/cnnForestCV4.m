function yPred = cnnForestCV4(train, XValid)
%
% Piotr Random forest on cnn for 4-class classification
    
    % Parameters combinations
    M = [25 50 70 100 120];
    maxDepth = [3 4 5 6 7 8];
    C = combine(M, maxDepth);
    
    % custom error function
    [CStar, ~, errors] = crossValid(train.X.cnn, train.y, 3, C, @applyRf, @BER);
    MStar = CStar(1);
    mdStar = CStar(2);
    
    % Predict
    yPred = applyRf(train.X.cnn, train.y, XValid.cnn, CStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'Forest on Hog - CV');
    contourf(M, maxDepth, ...
             reshape(errors, length(maxDepth), length(M))');
    colorbar;
    hold on;
    legend('ber error', 'Location', 'best');
    plot(MStar, mdStar, ...
         'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('number of tree');
    ylabel('maximum depth');
    hold off;
    
end

function yPred = applyRf(X, y, XValid, param)
%
    ops.M = param(1);
    ops.maxDepth = param(2);
    yPred = int8(forestComb4(X, y, XValid, ops)); 
end

