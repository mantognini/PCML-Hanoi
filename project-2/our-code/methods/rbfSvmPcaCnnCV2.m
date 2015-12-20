function yPred = rbfSvmPcaCnnCV2(train, XValid, category)
%
% Rbf svm on pca'd cnn using cross-validatoin for C, M
    
    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Find best C, M combination
    % C = 3.25, 150
    C = [8 9 10 11 12];
    M = 150;
    gammas = [1e-4 1.5e-4 2e-4 2.5e-4 3e-4];
    params = combine(C, gammas);
    f = @(X, y, XValid, params) svmPca2(X, y, XValid, M, @rbfKernel, params(1), params(2));
    [pStar, errStar, errors] = crossValid(train.X.cnn, train.y, 10, params, f, @BER);
    
    % Predict
    yPred = f(train.X.cnn, train.y, XValid.cnn, pStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'rbf-cnn-cv2');
    contourf(gammas, C, reshape(errors, length(C), length(gammas))); colorbar;
    hold on;
    legend('ber error', 'Location', 'best');
    plot(pStar(2), pStar(1), ...
         'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('gamma');
    ylabel('C');
    hold off;
    
end
