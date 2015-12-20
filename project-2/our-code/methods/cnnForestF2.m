function yPred = cnnForestF2(train, XValid, category)
%
% Piotr Random forest on cnn

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Parameters
    ops.M = 180;
    ops.maxDepth = 7;
    
    % Make y {1, 2} (Piotr needs it)
    yPred = forestComb2([train.X.cnn train.X.hog], train.y + 1, [XValid.cnn XValid.hog], ops) - 1; 
   
end
