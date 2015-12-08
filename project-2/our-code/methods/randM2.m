function yPred = randM2(train, XValid)
%
% Predict uniformly {0, 1}
    % As the 4-class random
    yPred = toBinary(randM4(train, XValid));
end