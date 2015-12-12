function yPred = rbfSvmPcaCnnCV2(train, XValid)
%
% Rbf svm on pca'd cnn using cross-validatoin for C, M
    
    % Make y binary
    train.y = toBinary(train.y);
    
    % Find best C, M combination
    % C = 3.25, 150
    C = linspace(2.5, 4, 5)';
    M = [25, 50, 75];
    params = combine(C, M);
    
    gamma = 0.00023;
    f = @(X, y, XValid, params) svmPca2(X, y, XValid, params(2), @rbfKernel, params(1), gamma);
    [pStar, errStar, errors] = crossValid(train.X.cnn, train.y, 1, params, f, @BER);
    
    % Predict
    yPred = f(train.X.cnn, train.y, XValid.cnn, pStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'rbf-cnn-cv2');
    contourf(M, C, reshape(errors, length(C), length(M))); colorbar;
    hold on;
    legend('ber error', 'Location', 'best');
    plot(pStar(2), pStar(1), ...
         'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('M');
    ylabel('C');
    hold off;
    
end
