function yPred = rbfSvmHogCV2p(train, XValid, label)
%
% rbf svm on hog feature using Cross-validation on C, gamma
    % Make y binary
    train.y = toBinary(train.y, label);
    
    % Find best C, gamma
    % gamma* range: 2 or 3
    % C* range: [2, 3] * 10^(-5)
    % (C*, gamma*) seems to be (2, 0.00023)
    gamma = linspace(0.5, 6, 4) * 10^(-4);
    C = linspace(1, 4, 4);
    params = combine(C, gamma);
    rbfSvm = @(trX, trY, XValid, params) svmF2(trX, trY, XValid, @rbfKernel, params(1), params(2));
    [pStar, ~, errors] = crossValid(train.X.hog, train.y, 4, params, rbfSvm, @BER);
    
    % Predict
    yPred = rbfSvm(train.X.hog, train.y, XValid.hog, pStar);
    
    % Optionally
    % Plot the cross-validation
    figure('Name', 'rbf hog, Cv2');
    contourf(gamma, C, reshape(errors, length(C), length(gamma))); colorbar;
    hold on;
    legend('ber error', 'Location', 'best');
    plot(pStar(2), pStar(1), ...
         'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('gamma');
    ylabel('C');
    hold off;
    
end