function yPred = hogForestCV2(train, XValid, category)
%
% Piotr Random forest on hog

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Parameters combinations
    M = 80:10:120;
    maxDepth = 4:1:8;
    C = combine(M, maxDepth);
    
    % custom error function
    [CStar, ~, errors] = crossValid(train.X.hog, train.y, 4, C, @applyRf, @BER);
    MStar = CStar(1);
    mdStar = CStar(2);
    
    % Predict
    yPred = applyRf(train.X.hog, train.y, XValid.hog, CStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'Forest on Hog - CV');
    contourf(M, maxDepth, ...
             reshape(errors, length(maxDepth), length(M)));
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
    
    % Make y {1, 2} (Piotr needs it)
    yPred = forestComb2(X, y + 1, XValid, ops) - 1; 
end

