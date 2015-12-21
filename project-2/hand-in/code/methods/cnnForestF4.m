function yPred = cnnForestF4(train, XValid)
%
% Piotr Random forest on cnn for 4-class classification
    
    % Parameters
    ops.M = 50;
    ops.maxDepth = 8;

    % Predict
    yPred = forestComb4(train.X.cnn, train.y, XValid.cnn, ops); 
    
end
