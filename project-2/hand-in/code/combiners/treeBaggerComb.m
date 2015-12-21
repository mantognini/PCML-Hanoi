function yPred = treeBaggerComb(X, y, XValid, numTree)
%
% Predict using treeBagger from matlab
    tb = TreeBagger(numTree, X, y);
    yPred = int8(cell2mat(predict(tb, XValid)) == '1');
end