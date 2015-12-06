function yPred = randomMethod(train, XValid)
    % As the 4-class random
    yPred = toBinary(randomMethod4(train, XValid));
end