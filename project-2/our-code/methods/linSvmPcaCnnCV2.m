function yPred = linSvmPcaCnnCV2(train, XValid)
%
% Linear svm on hog feature using Cross-validation on C

    % SETTINGS:
    M = 200; % might not be optimal
    
    % Apply PCA
    [TrZ, TeZ] = pcaCnn(M, train, XValid);

    % Make y binary
    train.y = toBinary(train.y);
    
    % Find best C
    C = linspace(0.00005, 0.0005, 20)';
    [CStar, errStar, errors] = crossValid(TrZ, train.y, 10, C, @linSvmF2, @BER);
    
    % Predict
    yPred = linSvmF2(TrZ, train.y, TeZ, CStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'Cross-valid, linear SVM, PCA CNN, Binary');
    plot(C, errors, 'm-o');
    hold on;
    legend('ber error', 'Location', 'best');
    plot(CStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('C');
    ylabel('BER');
    hold off;
    
end
