function y = toBinary(y)
    % 4-class into binary
    otherIdx = (y == 4);
    y(otherIdx) = 0;
    y(~otherIdx) = 1;
end