function yPred = linSvmHogCV2(train, XValid, category)
%
% Linear svm on hog feature using Cross-validation on C
    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Find best C
    C = linspace(0.00005, 0.0005, 10)';
    [CStar, errStar, errors] = crossValid(train.X.hog, train.y, 1, C, @linSvmF2, @BER);
    
    % Predict
    yPred = linSvmF2(train.X.hog, train.y, XValid.hog, CStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'Cross-valid, linear SVM, HOG, Binary');
    plot(C, errors, 'r-o');
    hold on;
    legend('ber error', 'Location', 'best');
    plot(CStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('C');
    ylabel('BER');
    hold off;
end