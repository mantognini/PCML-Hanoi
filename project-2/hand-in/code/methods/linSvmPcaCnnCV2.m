function yPred = linSvmPcaCnnCV2(train, XValid)
%
% Linear svm on pca'd cnn using cross-validatoin for C, M
    
    % Make y binary
    train.y = toBinary(train.y);
    
    % Find best C, M combination
    C = linspace(1, 4, 4) * 10^(-4);
    M = 1300;
    params = combine(C, M);
    f = @(X, y, XValid, params) svmPca2(X, y, XValid, params(2), @linearKernel, params(1), []);
    [pStar, errStar, errors] = crossValid(train.X.cnn, train.y, 5, params, f, @BER);
    
    % Predict
    yPred = f(train.X.cnn, train.y, XValid.cnn, pStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'linSvmPcaCnnCV2');
    plot(M, errors, 'm-o');
    hold on;
    legend('ber error', 'Location', 'best');
    plot(pStar(2), errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('M');
    ylabel('BER');
    hold off;
    
end
