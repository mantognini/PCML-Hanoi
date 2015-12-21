function yPred = hogForestF2(train, XValid, category)
%
% Piotr Random forest on hog

    % Make y binary
    train.y = toBinary(train.y, category);
    
    % Parameters
    ops.M = 110;
    ops.maxDepth = 5;
    
    % Make y {1, 2} (Piotr needs it)
    yPred = forestComb2(train.X.hog, train.y + 1, XValid.hog, ops) - 1; 
   
end
