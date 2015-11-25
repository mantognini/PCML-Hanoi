function [data] = loadClassificationData()
%LOADCLASSIFICATIONDATA Load data for classification
%
% The returned data is presented as a structure with the following
% interesting fields inside:
%  - train.X
%  - train.y
%  - test.X
%

    load('HaNoi_classification');
    
    fprintf('Converting -1 into 0 for simplification purposes.\n');
    y_train(y_train == -1) = 0;
    
    data.train.X = X_train;
    data.train.y = y_train;
    data.test.X  = X_test;

end

