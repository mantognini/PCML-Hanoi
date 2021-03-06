function yPred = pcaNnHogCV2(train, XValid)
%
% Apply PCA followed by NN on HOG feature

    % SETTINGS:
    M          = 500; % might be lower maybe
    EPOCHS     = 150;
    
    % Apply PCA
    [TrZ, TeZ] = pcaHog(M, train, XValid);
    
    % Apply NN
    myNN = @(TrX, y, TeX, c) nn(c(1), c(2), EPOCHS, 1, TrX, y, TeX);
    myBER = @(y4, yPred2) BER(toBinary(y4), yPred2);
    
    % Find best inner layer size
    innerSizes    = 10:5:150;
    learningRates = 0.1:0.1:4;
    
    C = combine(innerSizes, learningRates);
    K = 2; % no k-folding with NN
    [CStar, ~, errors] = crossValid(TrZ, train.y, K, C, myNN, myBER);
    innerSizeStar    = CStar(1);
    learningRateStar = CStar(2);
    
    % Predict
    yPred = nn(innerSizeStar, learningRateStar, EPOCHS, 1, TrZ, train.y, TeZ);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'Cross-valid, PCA HOG, NN, Binary');
    contourf(innerSizes, learningRates, ...
             reshape(errors, length(learningRates), length(innerSizes)));
    caxis([0.20 0.40]);
    colorbar;
    hold on;
    legend('ber error', 'Location', 'best');
    plot(innerSizeStar, learningRateStar, ...
         'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('inner layer size');
    ylabel('learning rate');
    hold off;
    
end

