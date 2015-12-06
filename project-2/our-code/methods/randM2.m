function yPred = randM2(train, XValid)
    % As the 4-class random
    yPred = toBinary(randM4(train, XValid));
end