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
    
    data.train.X = X_train;
    data.train.y = y_train;
    data.test.X  = X_test;

end

